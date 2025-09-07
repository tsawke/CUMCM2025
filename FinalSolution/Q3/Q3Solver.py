import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# constants
g = 9.81
EPS = 1e-12
DT_FINE = 0.01

FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0])
REAL_TARGET_PARAMS = {
    "center": np.array([0.0, 200.0, 0.0]),
    "radius": 7.0,
    "height": 10.0
}

FY1_INIT_POS = np.array([17800.0, 0.0, 1800.0])
FY1_SPEED_RANGE = [70.0, 140.0]

SMOKE_PARAMS = {
    "radius": 10.0,
    "sinkSpeed": 3.0,
    "effectTime": 20.0
}

MISSILE_M1_PARAMS = {
    "initPos": np.array([20000.0, 0.0, 2000.0]),
    "speed": 300.0,
}

missileDirection = (FAKE_TARGET_ORIGIN - MISSILE_M1_PARAMS["initPos"]) / np.linalg.norm(FAKE_TARGET_ORIGIN - MISSILE_M1_PARAMS["initPos"])
MISSILE_M1_PARAMS["direction"] = missileDirection
MISSILE_ARRIVAL_TIME = np.linalg.norm(FAKE_TARGET_ORIGIN - MISSILE_M1_PARAMS["initPos"]) / MISSILE_M1_PARAMS["speed"]

print(f"[Q3] Missile M1 arrival time: {MISSILE_ARRIVAL_TIME:.2f}s")


def GenerateTargetSamples(targetParams, thetaSamples = 40, heightSamples = 15):
    """Generate target sampling points"""
    samples = []
    centerXy = targetParams["center"][:2]
    baseZ = targetParams["center"][2]
    topZ = targetParams["center"][2] + targetParams["height"]
    radius = targetParams["radius"]
    
    thetaValues = np.linspace(0, 2*np.pi, thetaSamples, endpoint = False)
    
    for theta in thetaValues:
        x = centerXy[0] + radius * np.cos(theta)
        y = centerXy[1] + radius * np.sin(theta)
        samples.append([x, y, baseZ])
    
    for theta in thetaValues:
        x = centerXy[0] + radius * np.cos(theta)
        y = centerXy[1] + radius * np.sin(theta)
        samples.append([x, y, topZ])
    
    heightValues = np.linspace(baseZ, topZ, heightSamples)
    for z in heightValues:
        for theta in thetaValues:
            x = centerXy[0] + radius * np.cos(theta)
            y = centerXy[1] + radius * np.sin(theta)
            samples.append([x, y, z])
    
    interiorRadii = np.linspace(0, radius, 4)
    interiorHeights = np.linspace(baseZ, topZ, 8)
    interiorAngles = np.linspace(0, 2*np.pi, 16, endpoint = False)
    
    for z in interiorHeights:
        for r in interiorRadii:
            for theta in interiorAngles:
                x = centerXy[0] + r * np.cos(theta)
                y = centerXy[1] + r * np.sin(theta)
                samples.append([x, y, z])
    
    uniqueSamples = np.unique(np.array(samples), axis = 0)
    print(f"[Q3] Generated {len(uniqueSamples)} target points")
    return uniqueSamples


def LineSegmentSphereIntersection(pointM, pointP, sphereCenter, sphereRadius):
    """Check line-sphere intersection"""
    mpVector = pointP - pointM
    mcVector = sphereCenter - pointM
    
    a = np.dot(mpVector, mpVector)
    
    if a < EPS:
        return np.linalg.norm(mcVector) <= sphereRadius + EPS
    
    b = -2 * np.dot(mpVector, mcVector)
    c = np.dot(mcVector, mcVector) - sphereRadius**2
    discriminant = b**2 - 4*a*c
    
    if discriminant < -EPS:
        return False
    
    discriminant = max(discriminant, 0)
    sqrtDiscriminant = np.sqrt(discriminant)
    
    t1 = (-b - sqrtDiscriminant) / (2*a)
    t2 = (-b + sqrtDiscriminant) / (2*a)
    
    return (t1 <= 1.0 + EPS) and (t2 >= -EPS)


def CalSmokeShieldingIntervals(bombParams, targetSamplePoints):
    """Calculate shielding intervals"""
    theta, velocity, dropDelay, detonationDelay = bombParams
    
    uavDirection = np.array([np.cos(theta), np.sin(theta), 0.0])
    dropPosition = FY1_INIT_POS + velocity * dropDelay * uavDirection
    
    detonationXy = dropPosition[:2] + velocity * detonationDelay * uavDirection[:2]
    detonationZ = dropPosition[2] - 0.5 * g * detonationDelay**2
    
    if detonationZ < 0:
        return []
    
    detonationPosition = np.array([detonationXy[0], detonationXy[1], detonationZ])
    
    detonationTime = dropDelay + detonationDelay
    smokeStartTime = detonationTime
    smokeEndTime = min(detonationTime + SMOKE_PARAMS["effectTime"], MISSILE_ARRIVAL_TIME)
    
    if smokeStartTime >= smokeEndTime:
        return []
    
    timePoints = np.arange(smokeStartTime, smokeEndTime + DT_FINE, DT_FINE)
    shieldingIntervals = []
    currentlyShielded = False
    intervalStartTime = 0
    
    for currentTime in timePoints:
        missilePosition = MISSILE_M1_PARAMS["initPos"] + MISSILE_M1_PARAMS["speed"] * currentTime * MISSILE_M1_PARAMS["direction"]
        
        sinkDuration = currentTime - detonationTime
        smokeCenter = np.array([
            detonationPosition[0], 
            detonationPosition[1], 
            detonationPosition[2] - SMOKE_PARAMS["sinkSpeed"] * sinkDuration
        ])
        
        if smokeCenter[2] < 0:
            if currentlyShielded:
                shieldingIntervals.append([intervalStartTime, currentTime])
                currentlyShielded = False
            continue
        
        completeShielding = True
        for targetPoint in targetSamplePoints:
            if not LineSegmentSphereIntersection(missilePosition, targetPoint, 
                                               smokeCenter, SMOKE_PARAMS["radius"]):
                completeShielding = False
                break
        
        if completeShielding and not currentlyShielded:
            intervalStartTime = currentTime
            currentlyShielded = True
        elif not completeShielding and currentlyShielded:
            shieldingIntervals.append([intervalStartTime, currentTime])
            currentlyShielded = False
    
    if currentlyShielded:
        shieldingIntervals.append([intervalStartTime, smokeEndTime])
    
    return shieldingIntervals


def MergeTimeIntervals(intervalList):
    """Merge intervals"""
    if not intervalList:
        return 0.0, []
    
    sortedIntervals = sorted(intervalList, key = lambda x: x[0])
    mergedIntervals = [sortedIntervals[0]]
    
    for currentInterval in sortedIntervals[1:]:
        lastInterval = mergedIntervals[-1]
        if currentInterval[0] <= lastInterval[1] + EPS:
            mergedIntervals[-1] = [lastInterval[0], max(lastInterval[1], currentInterval[1])]
        else:
            mergedIntervals.append(currentInterval)
    
    totalDuration = sum([endTime - startTime for startTime, endTime in mergedIntervals])
    return totalDuration, mergedIntervals


def ObjectiveFunction(optimizationParams, targetSamplingPoints):
    """Objective function"""
    theta, velocity, t1_1, t2_1, deltaT2, t2_2, deltaT3, t2_3 = optimizationParams
    
    if not (FY1_SPEED_RANGE[0] - EPS <= velocity <= FY1_SPEED_RANGE[1] + EPS):
        return 0.0
    if deltaT2 < 1.0 - EPS or deltaT3 < 1.0 - EPS:
        return 0.0
    if any(param < -EPS for param in [t1_1, t2_1, t2_2, t2_3]):
        return 0.0
    
    t1_2 = t1_1 + deltaT2
    t1_3 = t1_2 + deltaT3
    
    bomb1Params = [theta, velocity, t1_1, t2_1]
    bomb2Params = [theta, velocity, t1_2, t2_2]
    bomb3Params = [theta, velocity, t1_3, t2_3]
    
    allIntervals = []
    allIntervals.extend(CalSmokeShieldingIntervals(bomb1Params, targetSamplingPoints))
    allIntervals.extend(CalSmokeShieldingIntervals(bomb2Params, targetSamplingPoints))
    allIntervals.extend(CalSmokeShieldingIntervals(bomb3Params, targetSamplingPoints))
    
    totalDuration, _ = MergeTimeIntervals(allIntervals)
    return totalDuration


class ParticleSwarmOptimizer:
    """PSO for smoke bomb deployment"""
    
    def __init__(self, objectiveFunc, paramBounds, numParticles = 50, maxIterations = 100):
        self.objectiveFunc = objectiveFunc
        self.bounds = paramBounds
        self.numParticles = numParticles
        self.maxIterations = maxIterations
        self.dimension = len(paramBounds)
        
        self.positions = np.zeros((numParticles, self.dimension))
        self.velocities = np.zeros((numParticles, self.dimension))
        
        for i in range(self.dimension):
            self.positions[:, i] = np.random.uniform(
                paramBounds[i][0], paramBounds[i][1], numParticles
            )
            velocityRange = paramBounds[i][1] - paramBounds[i][0]
            self.velocities[:, i] = 0.1 * np.random.uniform(
                -velocityRange, velocityRange, numParticles
            )
        
        self.personalBestPositions = self.positions.copy()
        self.personalBestFitness = np.array([self.objectiveFunc(pos) for pos in self.positions])
        
        self.globalBestIndex = np.argmax(self.personalBestFitness)
        self.globalBestPosition = self.personalBestPositions[self.globalBestIndex].copy()
        self.globalBestFitness = self.personalBestFitness[self.globalBestIndex]
        
        self.fitnessHistory = [self.globalBestFitness]
        
        print(f"[Q3] PSO initialized with {numParticles} particles")
        print(f"[Q3] Initial best fitness: {self.globalBestFitness:.4f}")
    
    def Optimize(self):
        """Run PSO optimization"""
        print("[Q3] Starting PSO optimization...")
        
        for iteration in range(self.maxIterations):
            inertiaWeight = 0.9 - 0.5 * (iteration / self.maxIterations)
            cognitiveFactor = 2.0
            socialFactor = 2.0
            
            for i in range(self.numParticles):
                currentFitness = self.objectiveFunc(self.positions[i])
                
                if currentFitness > self.personalBestFitness[i]:
                    self.personalBestFitness[i] = currentFitness
                    self.personalBestPositions[i] = self.positions[i].copy()
                
                if currentFitness > self.globalBestFitness:
                    self.globalBestFitness = currentFitness
                    self.globalBestPosition = self.positions[i].copy()
                
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)
                
                self.velocities[i] = (
                    inertiaWeight * self.velocities[i] +
                    cognitiveFactor * r1 * (self.personalBestPositions[i] - self.positions[i]) +
                    socialFactor * r2 * (self.globalBestPosition - self.positions[i])
                )
                
                self.positions[i] += self.velocities[i]
                
                for j in range(self.dimension):
                    if self.positions[i][j] < self.bounds[j][0]:
                        self.positions[i][j] = self.bounds[j][0]
                    elif self.positions[i][j] > self.bounds[j][1]:
                        self.positions[i][j] = self.bounds[j][1]
            
            self.fitnessHistory.append(self.globalBestFitness)
            
            if (iteration + 1) % 10 == 0:
                print(f"[Q3] Iteration {iteration+1:3d}/{self.maxIterations} | "
                      f"Best Fitness: {self.globalBestFitness:.6f} | "
                      f"Inertia Weight: {inertiaWeight:.3f}")
        
        print(f"[Q3] PSO completed. Final fitness: {self.globalBestFitness:.6f}")
        return self.globalBestPosition, self.globalBestFitness, self.fitnessHistory


def SaveStandardExcelFormat(optimalParams, totalShieldingDuration, bombIntervalsData, filename = "result3"):
    """Save Excel results"""
    thetaOpt, vOpt, t1_1Opt, t2_1Opt, deltaT2Opt, t2_2Opt, deltaT3Opt, t2_3Opt = optimalParams
    
    t1_2Opt = t1_1Opt + deltaT2Opt
    t1_3Opt = t1_2Opt + deltaT3Opt
    
    uavDirection = np.array([np.cos(thetaOpt), np.sin(thetaOpt), 0.0])
    
    excelData = []
    bombTimings = [(t1_1Opt, t2_1Opt), (t1_2Opt, t2_2Opt), (t1_3Opt, t2_3Opt)]
    
    for bombIndex, (dropTime, detonationDelay) in enumerate(bombTimings):
        dropPosition = FY1_INIT_POS + vOpt * dropTime * uavDirection
        detonationXy = dropPosition[:2] + vOpt * detonationDelay * uavDirection[:2]
        detonationZ = dropPosition[2] - 0.5 * g * detonationDelay**2
        detonationPosition = np.array([detonationXy[0], detonationXy[1], detonationZ])
        
        individualDuration = 0.0
        if bombIndex < len(bombIntervalsData) and bombIntervalsData[bombIndex]:
            individualDuration = sum([end - start for start, end in bombIntervalsData[bombIndex]])
        
        excelData.append({
            "无人机运动方向": f"{thetaOpt:.6f} rad",
            "无人机运动速度 (m/s)": round(vOpt, 6),
            "烟幕干扰弹编号": bombIndex + 1,
            "烟幕干扰弹投放点的x坐标 (m)": round(dropPosition[0], 6),
            "烟幕干扰弹投放点的y坐标 (m)": round(dropPosition[1], 6),
            "烟幕干扰弹投放点的z坐标 (m)": round(dropPosition[2], 6),
            "烟幕干扰弹起爆点的x坐标 (m)": round(detonationPosition[0], 6),
            "烟幕干扰弹起爆点的y坐标 (m)": round(detonationPosition[1], 6),
            "烟幕干扰弹起爆点的z坐标 (m)": round(detonationPosition[2], 6),
            "有效干扰时长 (s)": round(individualDuration, 2)
        })
    
    df = pd.DataFrame(excelData)
    excelFilename = f"{filename}.xlsx"
    df.to_excel(excelFilename, index = False, engine = "openpyxl")
    
    print(f"[Q3] Results saved to: {excelFilename}")
    return df, excelFilename


def SaveComprehensiveTxtReport(optimalParams, totalDuration, intervalsData, 
                             optimizationHistory, computationTime, filename = "Q3_results.txt"):
    """Save results to txt"""
    thetaOpt, vOpt, t1_1Opt, t2_1Opt, deltaT2Opt, t2_2Opt, deltaT3Opt, t2_3Opt = optimalParams
    t1_2Opt = t1_1Opt + deltaT2Opt
    t1_3Opt = t1_2Opt + deltaT3Opt
    
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.write("Q3 SMOKE BOMB OPTIMIZATION RESULTS\n")
        file.write("=" * 50 + "\n")
        file.write(f"Computation Time: {computationTime:.3f}s\n")
        file.write(f"Total Shielding Duration: {totalDuration:.6f}s\n")
        file.write(f"UAV Speed: {vOpt:.3f} m/s\n")
        file.write(f"UAV Heading: {np.degrees(thetaOpt):.2f} degrees\n\n")
        
        bombTimings = [(t1_1Opt, t2_1Opt), (t1_2Opt, t2_2Opt), (t1_3Opt, t2_3Opt)]
        for bombId, (dropTime, detDelay) in enumerate(bombTimings, 1):
            file.write(f"Bomb {bombId}: Drop={dropTime:.3f}s, Delay={detDelay:.3f}s\n")
        
        file.write(f"\nFinal Fitness: {optimizationHistory[-1]:.6f}s\n")
    
    print(f"[Q3] Report saved to: {filename}")
    return filename


def CreateConvergencePlot(fitnessHistory, savePath = "Q3_convergence.png"):
    """Create convergence plot"""
    plt.figure(figsize = (10, 6))
    
    iterations = range(len(fitnessHistory))
    plt.plot(iterations, fitnessHistory, 'b-', linewidth = 2, label = "Best Fitness")
    plt.fill_between(iterations, fitnessHistory, alpha = 0.3, color = 'lightblue')
    
    plt.xlabel("Iteration")
    plt.ylabel("Shielding Duration (s)")
    plt.title("Q3 PSO Convergence")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(savePath, dpi = 300, bbox_inches = 'tight')
    plt.close()
    print(f"[Q3] Plot saved to: {savePath}")


def main():
    """Main function"""
    print("Q3 SMOKE BOMB OPTIMIZATION")
    print("=" * 50)
    
    startTime = time.time()
    
    print("[Q3] Generating target points...")
    targetSamples = GenerateTargetSamples(REAL_TARGET_PARAMS, thetaSamples = 40, heightSamples = 12)
    
    print("[Q3] Setting up optimization...")
    optimizationBounds = [
        (0.0, 2*np.pi),
        (70.0, 140.0),
        (0.0, 50.0),
        (0.0, 15.0),
        (1.0, 25.0),
        (0.0, 15.0),
        (1.0, 25.0),
        (0.0, 15.0)
    ]
    
    print("[Q3] Running PSO...")
    psoOptimizer = ParticleSwarmOptimizer(
        objectiveFunc = lambda params: ObjectiveFunction(params, targetSamples),
        paramBounds = optimizationBounds,
        numParticles = 60,
        maxIterations = 100
    )
    
    optimalParams, bestFitness, fitnessHistory = psoOptimizer.Optimize()
    optimizationTime = time.time() - startTime
    
    print("[Q3] Processing results...")
    thetaOpt, vOpt, t1_1Opt, t2_1Opt, deltaT2Opt, t2_2Opt, deltaT3Opt, t2_3Opt = optimalParams
    t1_2Opt = t1_1Opt + deltaT2Opt
    t1_3Opt = t1_2Opt + deltaT3Opt
    
    bombParamsList = [
        [thetaOpt, vOpt, t1_1Opt, t2_1Opt],
        [thetaOpt, vOpt, t1_2Opt, t2_2Opt],
        [thetaOpt, vOpt, t1_3Opt, t2_3Opt]
    ]
    
    intervalsData = []
    for bombParams in bombParamsList:
        intervals = CalSmokeShieldingIntervals(bombParams, targetSamples)
        intervalsData.append(intervals)
    
    allIntervals = []
    for intervals in intervalsData:
        allIntervals.extend(intervals)
    totalDuration, mergedIntervals = MergeTimeIntervals(allIntervals)
    
    print("[Q3] Saving results...")
    df, excelFile = SaveStandardExcelFormat(
        optimalParams, totalDuration, intervalsData, filename = "result3"
    )
    
    txtFile = SaveComprehensiveTxtReport(
        optimalParams, totalDuration, intervalsData, fitnessHistory, optimizationTime
    )
    
    CreateConvergencePlot(fitnessHistory, savePath = "Q3_convergence.png")
    
    print(f"\n[Q3] RESULTS:")
    print(f"Time: {optimizationTime:.2f}s")
    print(f"Shielding: {totalDuration:.6f}s")
    print(f"Speed: {vOpt:.3f} m/s")
    print(f"Heading: {np.degrees(thetaOpt):.2f}°")
    print(f"Files: {excelFile}, {txtFile}")
    
    return {
        "totalDuration": totalDuration,
        "optimizationTime": optimizationTime,
        "optimalParams": optimalParams,
        "excelFile": excelFile,
        "txtFile": txtFile
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n[Q3] Final: {results['totalDuration']:.6f}s")
    except Exception as e:
        print(f"[Q3] Error: {str(e)}")
        import traceback
        traceback.print_exc()