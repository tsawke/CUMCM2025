import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.optimize import differential_evolution, linear_sum_assignment

# Global Parameters
TRUE_TARGET = {
    "r": 7,
    "h": 10,
    "center": np.array([0, 200, 0]),
    "sample_points": None
}
MISSILES = {
    "M1": {"init_pos": np.array([20000, 0, 2000]), "dir": None, "flight_time": None},
    "M2": {"init_pos": np.array([19000, 600, 2100]), "dir": None, "flight_time": None},
    "M3": {"init_pos": np.array([18000, -600, 1900]), "dir": None, "flight_time": None}
}
MISSILE_SPEED = 300
G = 9.8
SMOKE_RADIUS = 10
SMOKE_SINK_SPEED = 3
SMOKE_EFFECTIVE_TIME = 20
DRONES = {
    "FY1": {"init_pos": np.array([17800, 0, 1800]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY2": {"init_pos": np.array([12000, 1400, 1400]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY3": {"init_pos": np.array([6000, -3000, 700]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY4": {"init_pos": np.array([11000, 2000, 1800]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False},
    "FY5": {"init_pos": np.array([13000, -2000, 1300]), "max_smoke": 3, "speed_range": [70, 140], 
            "smokes": [], "speed": None, "direction": None, "optimized": False}
}
DROP_INTERVAL = 1
TIME_STEP = 0.1

def GenerateTrueTargetSamples():
    samples = []
    r, h, center = TRUE_TARGET["r"], TRUE_TARGET["h"], TRUE_TARGET["center"]
    samples.append(center)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        samples.append(np.array([x, y, center[2]]))
    topCenter = center + np.array([0, 0, h])
    samples.append(topCenter)
    for theta in np.linspace(0, 2 * np.pi, 15):
        x = topCenter[0] + r * np.cos(theta)
        y = topCenter[1] + r * np.sin(theta)
        samples.append(np.array([x, y, topCenter[2]]))
    for z in np.linspace(center[2], topCenter[2], 5):
        for theta in np.linspace(0, 2 * np.pi, 12):
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            samples.append(np.array([x, y, z]))
    TRUE_TARGET["sample_points"] = np.array(samples)

def InitMissiles():
    for mName, mData in MISSILES.items():
        initPos = mData["init_pos"]
        dirVec = -initPos / np.linalg.norm(initPos)
        mData["dir"] = dirVec * MISSILE_SPEED
        mData["flight_time"] = np.linalg.norm(initPos) / MISSILE_SPEED
GenerateTrueTargetSamples()
InitMissiles()

# Core Functions
def GetMissilePos(mName, t):
    mData = MISSILES[mName]
    if t > mData["flight_time"]:
        return mData["init_pos"] + mData["dir"] * mData["flight_time"]
    return mData["init_pos"] + mData["dir"] * t

def GetDronePos(droneName, t):
    drone = DRONES[droneName]
    if drone["speed"] is None or drone["direction"] is None:
        return drone["init_pos"]
    
    vVec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    return drone["init_pos"] + vVec * t
def GetSmokePos(droneName, dropTime, detDelay, t):
    drone = DRONES[droneName]
    
    if t < dropTime:
        return None
    
    dropPos = GetDronePos(droneName, dropTime)
    
    if t < dropTime + detDelay:
        deltaT = t - dropTime
        vVec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                         drone["speed"] * np.sin(drone["direction"]), 0])
        x = dropPos[0] + vVec[0] * deltaT
        y = dropPos[1] + vVec[1] * deltaT
        z = dropPos[2] - 0.5 * G * deltaT ** 2
        return np.array([x, y, max(z, 0.1)])
    
    detTime = dropTime + detDelay
    if t > detTime + SMOKE_EFFECTIVE_TIME:
        return None
    
    deltaTDet = detDelay
    vVec = np.array([drone["speed"] * np.cos(drone["direction"]), 
                     drone["speed"] * np.sin(drone["direction"]), 0])
    detX = dropPos[0] + vVec[0] * deltaTDet
    detY = dropPos[1] + vVec[1] * deltaTDet
    detZ = dropPos[2] - 0.5 * G * deltaTDet ** 2
    
    if detZ < 0:
        detZ = 0.1
    
    deltaTAfter = t - detTime
    z = detZ - SMOKE_SINK_SPEED * deltaTAfter
    return np.array([detX, detY, max(z, 0.1)])
def SegmentSphereIntersect(p1, p2, center, radius):
    vecP = p2 - p1
    vecC = center - p1
    t = np.dot(vecC, vecP) / (np.dot(vecP, vecP) + 1e-8)
    
    if 0 <= t <= 1:
        nearest = p1 + t * vecP
    else:
        nearest = p1 if t < 0 else p2
    
    return np.linalg.norm(nearest - center) <= radius + 1e-8
def CalcSmokeEffectiveTime(droneName, mName, dropTime, detDelay):
    drone = DRONES[droneName]
    v, theta = drone["speed"], drone["direction"]
    
    if v is None or theta is None:
        return -1000
    
    if not (drone["speed_range"][0] - 1e-3 <= v <= drone["speed_range"][1] + 1e-3):
        return -1000
    
    detTime = dropTime + detDelay
    dropPos = GetDronePos(droneName, dropTime)
    deltaTDet = detDelay
    detZ = dropPos[2] - 0.5 * G * deltaTDet ** 2
    if detZ < -0.5:
        return -1000
    
    for smoke in drone["smokes"]:
        if abs(dropTime - smoke["drop_time"]) < DROP_INTERVAL - 0.1:
            return -1000
    
    maxT = min(detTime + SMOKE_EFFECTIVE_TIME, MISSILES[mName]["flight_time"] + 1)
    minT = max(detTime, 0)
    if minT >= maxT - 1e-3:
        return 0
    
    effectiveDuration = 0
    for t in np.arange(minT, maxT, TIME_STEP):
        mPos = GetMissilePos(mName, t)
        smokePos = GetSmokePos(droneName, dropTime, detDelay, t)
        if smokePos is None:
            continue
        
        allIntersect = True
        for sample in TRUE_TARGET["sample_points"]:
            if not SegmentSphereIntersect(mPos, sample, smokePos, SMOKE_RADIUS):
                allIntersect = False
                break
        if allIntersect:
            effectiveDuration += TIME_STEP
    return effectiveDuration

# Optimization Functions
def OptimizeSingleSmoke(droneName, mName):
    drone = DRONES[droneName]
    vMin, vMax = drone["speed_range"]
    maxFlightTime = MISSILES[mName]["flight_time"]
    
    bounds = [
        (vMin * 0.8, vMax * 1.2),
        (0, 2 * np.pi),
        (0, maxFlightTime - 1),
        (0.1, 20)
    ]
    
    def objective(x):
        v, theta, dropTime, detDelay = x
        drone["speed"] = v
        drone["direction"] = theta
        return -CalcSmokeEffectiveTime(droneName, mName, dropTime, detDelay)
    
    result = differential_evolution(
        func = objective,
        bounds = bounds,
        mutation = 0.8,
        recombination = 0.9,
        popsize = 60,
        maxiter = 80,
        tol = 1e-3,
        disp = False,
        polish = True
    )
    
    vOpt, thetaOpt, dropTimeOpt, detDelayOpt = result.x
    vOpt = np.clip(vOpt, vMin, vMax)
    effectiveTime = -result.fun
    return {
        "v": vOpt,
        "theta": thetaOpt,
        "drop_time": dropTimeOpt,
        "det_delay": detDelayOpt,
        "det_time": dropTimeOpt + detDelayOpt,
        "det_pos": GetSmokePos(droneName, dropTimeOpt, detDelayOpt, dropTimeOpt + detDelayOpt),
        "effective_time": effectiveTime if effectiveTime > 1e-3 else 0,
        "missile": mName
    }

def OptimizeDroneTrajectory(droneName, mName, retry = 0):
    drone = DRONES[droneName]
    vMin, vMax = drone["speed_range"]
    maxSmoke = drone["max_smoke"]
    
    vCandidates = np.linspace(vMin, vMax, 8)
    bestV = None
    bestSmokes = []
    maxTotalTime = 0
    
    for v in vCandidates:
        drone["speed"] = v
        tempSmokes = []
        
        for i in range(maxSmoke):
            minDropTime = tempSmokes[-1]["drop_time"] + DROP_INTERVAL if tempSmokes else 0
            maxDropTime = MISSILES[mName]["flight_time"] - 0.1
            if minDropTime >= maxDropTime - 1e-3:
                break
            
            def objective(x):
                theta, dropTime, detDelay = x
                drone["direction"] = theta
                return -CalcSmokeEffectiveTime(droneName, mName, dropTime, detDelay)
            
            result = differential_evolution(
                func = objective, 
                bounds = [(0, 2 * np.pi), (minDropTime, maxDropTime), (0.1, 10)],
                mutation = 0.7, 
                recombination = 0.8, 
                popsize = 50,
                maxiter = 60,
                disp = False
            )
            
            thetaOpt, dropTimeOpt, detDelayOpt = result.x
            drone["direction"] = thetaOpt
            effectiveTime = CalcSmokeEffectiveTime(droneName, mName, dropTimeOpt, detDelayOpt)
            
            if effectiveTime > 0.1:
                smoke = {
                    "v": v,
                    "theta": thetaOpt,
                    "drop_time": dropTimeOpt,
                    "det_delay": detDelayOpt,
                    "det_time": dropTimeOpt + detDelayOpt,
                    "det_pos": GetSmokePos(droneName, dropTimeOpt, detDelayOpt, dropTimeOpt + detDelayOpt),
                    "effective_time": effectiveTime,
                    "missile": mName
                }
                tempSmokes.append(smoke)
        
        totalTime = sum([s["effective_time"] for s in tempSmokes]) if tempSmokes else 0
        if totalTime > maxTotalTime:
            maxTotalTime = totalTime
            bestV = v
            bestSmokes = tempSmokes
    
    if not bestSmokes and retry < 3:
        print(f"[{droneName}] Optimization failed, retrying {retry + 1}/3...")
        return OptimizeDroneTrajectory(droneName, mName, retry + 1)
    
    if bestSmokes:
        dropPoints = []
        weights = []
        for smoke in bestSmokes:
            vVec = np.array([smoke["v"] * np.cos(smoke["theta"]), smoke["v"] * np.sin(smoke["theta"]), 0])
            dropPos = drone["init_pos"] + vVec * smoke["drop_time"]
            dropPoints.append(dropPos[:2])
            weights.append(smoke["effective_time"])
        dropPoints = np.array(dropPoints)
        weights = np.array(weights)
        
        X = np.column_stack([dropPoints[:, 0], np.ones(len(dropPoints))])
        W = np.diag(weights)
        try:
            k, b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ dropPoints[:, 1])
            refTheta = np.arctan(k)
        except np.linalg.LinAlgError:
            refTheta = np.mean([s["theta"] for s in bestSmokes]) if bestSmokes else 0
        
        for i, smoke in enumerate(bestSmokes):
            thetaCandidates = [refTheta - np.pi / 24, refTheta, refTheta + np.pi / 24]
            dropCandidates = [smoke["drop_time"] - 0.8, smoke["drop_time"], smoke["drop_time"] + 0.8]
            
            bestEffect = smoke["effective_time"]
            bestParams = (smoke["theta"], smoke["drop_time"])
            
            for theta in thetaCandidates:
                for dropTime in dropCandidates:
                    prevDropTime = bestSmokes[i - 1]["drop_time"] if i > 0 and i - 1 < len(bestSmokes) else -np.inf
                    if dropTime < prevDropTime + DROP_INTERVAL - 0.1:
                        continue
                    drone["direction"] = theta
                    effect = CalcSmokeEffectiveTime(droneName, mName, dropTime, smoke["det_delay"])
                    if effect > bestEffect:
                        bestEffect = effect
                        bestParams = (theta, dropTime)
            
            smoke["theta"], smoke["drop_time"] = bestParams
            smoke["det_time"] = smoke["drop_time"] + smoke["det_delay"]
            smoke["det_pos"] = GetSmokePos(droneName, smoke["drop_time"], smoke["det_delay"], smoke["det_time"])
            smoke["effective_time"] = bestEffect
    
    drone["speed"] = bestV
    drone["direction"] = refTheta if bestSmokes else None
    drone["smokes"] = bestSmokes
    return bestSmokes

# Task Assignment and Optimization
def AssignTasks(unoptimizedDrones = None):
    if unoptimizedDrones is None:
        unoptimizedDrones = list(DRONES.keys())
    
    missileList = list(MISSILES.keys())
    nDrones = len(unoptimizedDrones)
    nMissiles = len(missileList)
    if nDrones == 0:
        return {m: [] for m in missileList}
    
    costMatrix = np.zeros((nDrones, nMissiles))
    for i, dName in enumerate(unoptimizedDrones):
        dInit = DRONES[dName]["init_pos"]
        dAvgV = (DRONES[dName]["speed_range"][0] + DRONES[dName]["speed_range"][1]) / 2
        for j, mName in enumerate(missileList):
            mInit = MISSILES[mName]["init_pos"]
            mFlightTime = MISSILES[mName]["flight_time"]
            
            dist = np.linalg.norm(dInit - mInit)
            cost1 = dist / dAvgV
            cost2 = 1000 / (mFlightTime + 1)
            cost3 = abs(dAvgV - MISSILE_SPEED) / 100
            
            costMatrix[i][j] = cost1 + cost2 + cost3
    
    rowInd, colInd = linear_sum_assignment(costMatrix)
    assignments = {m: [] for m in missileList}
    for i, j in zip(rowInd, colInd):
        assignments[missileList[j]].append(unoptimizedDrones[i])
    
    assignedDrones = set(rowInd)
    for i in range(nDrones):
        if i not in assignedDrones:
            minCostJ = np.argmin(costMatrix[i])
            assignments[missileList[minCostJ]].append(unoptimizedDrones[i])
    
    return assignments

def IterativeOptimization(maxIterations = 20, improvementThreshold = 0.3, maxStallIter = 3):
    for dName in DRONES:
        DRONES[dName]["optimized"] = False
        DRONES[dName]["smokes"] = []
        DRONES[dName]["speed"] = None
        DRONES[dName]["direction"] = None
    
    allSmokes = []
    prevTotalTime = 0
    stallCount = 0
    
    for iteration in range(maxIterations):
        print(f"\n===== Iteration {iteration + 1}/{maxIterations} =====")
        
        dronesWithoutSolution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
        print(f"UAVs without valid solutions: {dronesWithoutSolution}")
        
        if not dronesWithoutSolution:
            print("All UAVs have found valid solutions, stopping iteration")
            break
        
        assignments = AssignTasks(dronesWithoutSolution)
        
        currentTotalTime = 0
        iterationSmokes = []
        optimizedThisIter = []
        
        for mName, droneNames in assignments.items():
            for dName in droneNames:
                if len(DRONES[dName]["smokes"]) > 0:
                    continue
                    
                print(f"Optimizing UAV {dName} to interfere with {mName}...")
                smokes = OptimizeDroneTrajectory(dName, mName)
                
                if smokes:
                    droneSmokes = [{**smoke, "drone": dName} for smoke in smokes]
                    iterationSmokes.extend(droneSmokes)
                    currentTotalTime += sum([s["effective_time"] for s in smokes])
                    print(f"[{dName}] Optimization successful: {len(smokes)} smoke bombs, total shielding duration {currentTotalTime:.2f}s")
                else:
                    print(f"[{dName}] Still no valid deployment strategy found, will continue trying in next iteration")
                
                DRONES[dName]["optimized"] = True
                optimizedThisIter.append(dName)
        
        allSmokes.extend(iterationSmokes)
        
        totalEffectiveTime = sum([sum([s["effective_time"] for s in d["smokes"]]) for d in DRONES.values()])
        print(f"Current total shielding duration: {totalEffectiveTime:.2f}s")
        print(f"UAVs optimized this round: {optimizedThisIter}")
        print(f"UAVs with valid solutions count: {len(DRONES) - len(dronesWithoutSolution)}/{len(DRONES)}")
        
        improvement = totalEffectiveTime - prevTotalTime
        print(f"Improvement from previous round: {improvement:.2f}s")
        
        if improvement < improvementThreshold:
            stallCount += 1
            print(f"Consecutive non-improvement count: {stallCount}/{maxStallIter}")
            if stallCount >= maxStallIter:
                if dronesWithoutSolution:
                    print(f"Consecutive {maxStallIter} rounds without effective improvement, but UAVs still need solutions, continuing optimization...")
                    stallCount = maxStallIter - 1
                else:
                    print(f"Consecutive {maxStallIter} rounds without effective improvement, stopping iteration")
                    break
        else:
            stallCount = 0
        
        prevTotalTime = totalEffectiveTime
    
    remainingWithoutSolution = [d for d in DRONES if len(DRONES[d]["smokes"]) == 0]
    if remainingWithoutSolution:
        print(f"\nWarning: Maximum iteration count reached, UAVs still without valid solutions: {remainingWithoutSolution}")
    
    return allSmokes

# Result Output and Visualization
def SaveResult(smokes, filename = "smoke_optimization_result.xlsx"):
    data = []
    for i, smoke in enumerate(smokes, 1):
        detPos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0, 0, 0])
        data.append({
            "Smoke_Bomb_Number": f"S{i}",
            "UAV_Number": smoke["drone"],
            "Speed_m_per_s": round(smoke["v"], 2),
            "Direction_degrees": round(np.degrees(smoke["theta"]), 2),
            "Drop_Time_s": round(smoke["drop_time"], 2),
            "Detonation_Delay_s": round(smoke["det_delay"], 2),
            "Detonation_Time_s": round(smoke["det_time"], 2),
            "Detonation_Point_X_m": round(detPos[0], 2),
            "Detonation_Point_Y_m": round(detPos[1], 2),
            "Detonation_Point_Z_m": round(detPos[2], 2),
            "Target_Missile": smoke["missile"],
            "Effective_Shielding_Duration_s": round(smoke["effective_time"], 2)
        })
    df = pd.DataFrame(data)
    df.to_excel(filename, index = False, engine = "openpyxl")
    print(f"Results saved to {filename}")
    return df

def VisualizeResult(smokes):
    if not smokes:
        print("No valid data for visualization")
        return
    
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16, 12))
    
    theta = np.linspace(0, 2 * np.pi, 100)
    xCircle = TRUE_TARGET["center"][0] + TRUE_TARGET["r"] * np.cos(theta)
    yCircle = TRUE_TARGET["center"][1] + TRUE_TARGET["r"] * np.sin(theta)
    ax1.plot(xCircle, yCircle, "r-", label = "Real Target Projection")
    ax1.scatter(TRUE_TARGET["center"][0], TRUE_TARGET["center"][1], c = "r", marker = "*", s = 200, label = "Real Target Center")
    
    colors = ["red", "green", "blue"]
    for i, (mName, mData) in enumerate(MISSILES.items()):
        tRange = np.linspace(0, mData["flight_time"], 100)
        posList = [GetMissilePos(mName, t)[:2] for t in tRange]
        posArr = np.array(posList)
        ax1.plot(posArr[:, 0], posArr[:, 1], f"{colors[i]}--", label = f"{mName} Trajectory")
        ax1.scatter(mData["init_pos"][0], mData["init_pos"][1], c = colors[i], s = 100, label = f"{mName} Initial Position")
    
    droneColors = ["orange", "purple", "cyan", "magenta", "brown"]
    for i, (dName, dData) in enumerate(DRONES.items()):
        if not dData["smokes"]:
            continue
        lastSmoke = dData["smokes"][-1] if dData["smokes"] else None
        if lastSmoke:
            tRange = np.linspace(0, lastSmoke["drop_time"], 50)
            vVec = np.array([dData["speed"] * np.cos(dData["direction"]), dData["speed"] * np.sin(dData["direction"]), 0]) if dData["speed"] and dData["direction"] else np.array([0, 0, 0])
            posList = [dData["init_pos"] + vVec * t for t in tRange]
            posArr = np.array(posList)
            ax1.plot(posArr[:, 0], posArr[:, 1], f"{droneColors[i]}-", label = f"{dName} Trajectory")
            ax1.scatter(dData["init_pos"][0], dData["init_pos"][1], c = droneColors[i], s = 100, marker = "^", label = f"{dName} Initial Position")
            
            for smoke in dData["smokes"]:
                detPos = smoke["det_pos"]
                if detPos is not None:
                    ax1.scatter(detPos[0], detPos[1], c = droneColors[i], s = 50, alpha = 0.7)
                    circle = plt.Circle((detPos[0], detPos[1]), SMOKE_RADIUS, color = droneColors[i], alpha = 0.2)
                    ax1.add_patch(circle)
    
    ax1.set_xlabel("X Coordinate (m)")
    ax1.set_ylabel("Y Coordinate (m)")
    ax1.set_title("UAV, Missile Trajectories and Smoke Detonation Points")
    ax1.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")
    ax1.grid(True, alpha = 0.3)
    
    missileEffect = {m: 0 for m in MISSILES.keys()}
    for smoke in smokes:
        missileEffect[smoke["missile"]] += smoke["effective_time"]
    ax2.bar(missileEffect.keys(), missileEffect.values(), color = colors)
    ax2.set_xlabel("Missile ID")
    ax2.set_ylabel("Total Shielding Duration (s)")
    ax2.set_title("Total Shielding Duration per Missile")
    for m, t in missileEffect.items():
        ax2.text(m, t + 0.5, f"{t:.1f}s", ha = "center")
    
    droneSmokeCount = {d: len(DRONES[d]["smokes"]) for d in DRONES.keys()}
    ax3.bar(droneSmokeCount.keys(), droneSmokeCount.values(), color = droneColors)
    ax3.set_xlabel("UAV ID")
    ax3.set_ylabel("Number of Smoke Bombs")
    ax3.set_title("Smoke Bomb Deployment Count per UAV")
    for d, cnt in droneSmokeCount.items():
        ax3.text(d, cnt + 0.05, str(cnt), ha = "center")
    
    effectTimes = [smoke["effective_time"] for smoke in smokes]
    ax4.hist(effectTimes, bins = 10, color = "skyblue", edgecolor = "black", alpha = 0.7)
    ax4.set_xlabel("Single Smoke Bomb Shielding Duration (s)")
    ax4.set_ylabel("Number of Smoke Bombs")
    ax4.set_title("Single Smoke Bomb Shielding Duration Distribution")
    ax4.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig("smoke_optimization_visualization.png", dpi = 300, bbox_inches = "tight")
    plt.close()

def SaveComprehensiveReport(smokes, filename = "optimization_comprehensive_report.txt"):
    
    with open(filename, 'w', encoding = 'utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MULTI-UAV SMOKE BOMB DEPLOYMENT OPTIMIZATION - COMPREHENSIVE REPORT\n")
        f.write("Problem 5: 5 UAVs (FY1-FY5) vs 3 Missiles (M1, M2, M3)\n")
        f.write("Algorithm: Enhanced Iterative Differential Evolution\n")
        f.write("=" * 100 + "\n")
        f.write(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Smoke Bombs Deployed: {len(smokes)}\n")
        f.write(f"Total System Effectiveness: {sum([s['effective_time'] for s in smokes]):.6f} seconds\n")
        f.write("\n")
        
        f.write("PROBLEM CONFIGURATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Real Target: Cylindrical target at {TRUE_TARGET['center']} with radius {TRUE_TARGET['r']}m, height {TRUE_TARGET['h']}m\n")
        f.write(f"Fake Target: Point at origin [0, 0, 0] (missile destination)\n")
        f.write(f"Missiles:\n")
        for mName, mData in MISSILES.items():
            f.write(f"  {mName}: Initial position {mData['init_pos']}, flight time {mData['flight_time']:.2f}s\n")
        f.write(f"UAVs:\n")
        for dName, dData in DRONES.items():
            f.write(f"  {dName}: Initial position {dData['init_pos']}, speed range {dData['speed_range']} m/s\n")
        f.write(f"Smoke Parameters: Radius {SMOKE_RADIUS}m, sink speed {SMOKE_SINK_SPEED}m/s, duration {SMOKE_EFFECTIVE_TIME}s\n")
        f.write("\n")
        
        f.write("DETAILED DEPLOYMENT STRATEGY:\n")
        f.write("-" * 50 + "\n")
        
        for dName, dData in DRONES.items():
            f.write(f"UAV {dName}:\n")
            if dData["smokes"]:
                uavTotal = sum([s["effective_time"] for s in dData["smokes"]])
                f.write(f"  Status: ACTIVE DEPLOYMENT\n")
                f.write(f"  Optimized Speed: {dData['speed']:.3f} m/s\n")
                f.write(f"  Optimized Direction: {dData['direction']:.6f} rad ({np.degrees(dData['direction']):.2f}°)\n")
                f.write(f"  Number of Smoke Bombs: {len(dData['smokes'])}\n")
                f.write(f"  Total Effectiveness: {uavTotal:.6f} s\n")
                
                for j, smoke in enumerate(dData["smokes"], 1):
                    detPos = smoke["det_pos"] if smoke["det_pos"] is not None else np.array([0, 0, 0])
                    f.write(f"    Smoke Bomb {j}:\n")
                    f.write(f"      Drop Time: {smoke['drop_time']:.3f}s\n")
                    f.write(f"      Detonation Delay: {smoke['det_delay']:.3f}s\n")
                    f.write(f"      Detonation Position: ({detPos[0]:.2f}, {detPos[1]:.2f}, {detPos[2]:.2f}) m\n")
                    f.write(f"      Effectiveness: {smoke['effective_time']:.6f}s\n")
                    f.write(f"      Target Missile: {smoke['missile']}\n")
            else:
                f.write(f"  Status: NO VALID DEPLOYMENT FOUND\n")
            f.write("\n")
        
        totalEffectiveness = sum([s["effective_time"] for s in smokes])
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total System Effectiveness: {totalEffectiveness:.6f} seconds\n")
        f.write(f"Average Effectiveness per Bomb: {totalEffectiveness / len(smokes):.6f} seconds\n")
        
        missileCoverage = {m: 0 for m in MISSILES.keys()}
        for smoke in smokes:
            missileCoverage[smoke["missile"]] += smoke["effective_time"]
        
        f.write(f"\nMissile Coverage Analysis:\n")
        for mName, coverageTime in missileCoverage.items():
            flightTime = MISSILES[mName]["flight_time"]
            coveragePercentage = coverageTime / flightTime * 100
            f.write(f"  {mName}: {coverageTime:.3f}s shielding ({coveragePercentage:.1f}% of flight time)\n")
        
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("END OF COMPREHENSIVE REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"Comprehensive report saved to {filename}")
    return filename

# Main Execution
if __name__ == "__main__":
    allSmokes = IterativeOptimization(maxIterations = 20, improvementThreshold = 0.3, maxStallIter = 3)
    
    if allSmokes:
        resultDf = SaveResult(allSmokes)
        VisualizeResult(allSmokes)
        comprehensiveReportFile = SaveComprehensiveReport(allSmokes)
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY:")
        print(f"Total smoke bomb count: {len(allSmokes)}")
        print(f"Total shielding duration: {sum([s['effective_time'] for s in allSmokes]):.2f}s")
        print("\nUAV deployment details:")
        for dName, dData in DRONES.items():
            if dData["smokes"]:
                total = sum([s["effective_time"] for s in dData["smokes"]])
                print(f"{dName}: {len(dData['smokes'])} bombs, total shielding duration {total:.2f}s")
            else:
                print(f"{dName}: No valid deployment strategy found")
        print("=" * 50)
    else:
        print("No valid smoke bomb deployment strategy found")