#!/usr/bin/env python3
"""Q3Checker - High precision Q3 verifier"""

import math
import numpy as np
import pandas as pd
import argparse
import time
from typing import List, Tuple, Optional, Dict, Any

# constants
G = 9.8
CLOUD_RADIUS = 10.0
CLOUD_SINK_SPEED = 3.0
CLOUD_EFFECT_TIME = 20.0
MISSILE_SPEED = 300.0

FAKE_TARGET = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_BASE_CENTER = np.array([0.0, 200.0, 0.0])
CYLINDER_RADIUS = 7.0
CYLINDER_HEIGHT = 10.0

M1_INIT = np.array([20000.0, 0.0, 2000.0])
FY1_INIT = np.array([17800.0, 0.0, 1800.0])

MIN_DROP_INTERVAL = 1.0
UAV_SPEED_MIN = 70.0
UAV_SPEED_MAX = 140.0

def MissilePosition(t: float) -> np.ndarray:
    """Calculate missile position at time t"""
    if t < 0:
        return M1_INIT.copy()
    
    direction = (FAKE_TARGET - M1_INIT) / np.linalg.norm(FAKE_TARGET - M1_INIT)
    position = M1_INIT + MISSILE_SPEED * t * direction
    
    if np.linalg.norm(position - FAKE_TARGET) < 1.0:
        return FAKE_TARGET.copy()
    
    return position

def UavPosition(t: float, speed: float, heading: float) -> np.ndarray:
    """Calculate UAV position at time t"""
    if t < 0:
        return FY1_INIT.copy()
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    return np.array([
        FY1_INIT[0] + vx * t,
        FY1_INIT[1] + vy * t,
        FY1_INIT[2]
    ])

def SmokeBombTrajectory(tDrop: float, fuseDelay: float, speed: float, heading: float, t: float) -> Optional[np.ndarray]:
    """Calculate smoke bomb position at time t"""
    if t < tDrop:
        return None
    
    tExplosion = tDrop + fuseDelay
    if t >= tExplosion:
        return None
    
    dropPos = UavPosition(tDrop, speed, heading)
    dt = t - tDrop
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    x = dropPos[0] + vx * dt
    y = dropPos[1] + vy * dt
    z = dropPos[2] - 0.5 * G * dt * dt
    
    return np.array([x, y, z])

def ExplosionPosition(tDrop: float, fuseDelay: float, speed: float, heading: float) -> np.ndarray:
    """Calculate explosion position"""
    dropPos = UavPosition(tDrop, speed, heading)
    
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    
    x = dropPos[0] + vx * fuseDelay
    y = dropPos[1] + vy * fuseDelay
    z = dropPos[2] - 0.5 * G * fuseDelay * fuseDelay
    
    return np.array([x, y, z])

def CloudCenterPosition(explosionPos: np.ndarray, tExplosion: float, t: float) -> Optional[np.ndarray]:
    """Calculate cloud center position at time t"""
    if t < tExplosion:
        return None
    
    if t > tExplosion + CLOUD_EFFECT_TIME:
        return None
    
    sinkDistance = CLOUD_SINK_SPEED * (t - tExplosion)
    centerZ = explosionPos[2] - sinkDistance
    
    if centerZ < CLOUD_RADIUS:
        return None
    
    return np.array([explosionPos[0], explosionPos[1], centerZ])

def GenerateCylinderSamplingPoints(nCircumference: int = 36, nHeight: int = 5) -> np.ndarray:
    """Generate cylinder surface sampling points"""
    points = []
    
    for i in range(nCircumference):
        angle = 2 * math.pi * i / nCircumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2]
        points.append([x, y, z])
    
    for i in range(nCircumference):
        angle = 2 * math.pi * i / nCircumference
        x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
        y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT
        points.append([x, y, z])
    
    for k in range(1, nHeight - 1):
        heightRatio = k / (nHeight - 1)
        z = TRUE_TARGET_BASE_CENTER[2] + CYLINDER_HEIGHT * heightRatio
        
        nSide = nCircumference // 2
        for i in range(nSide):
            angle = 2 * math.pi * i / nSide
            x = TRUE_TARGET_BASE_CENTER[0] + CYLINDER_RADIUS * math.cos(angle)
            y = TRUE_TARGET_BASE_CENTER[1] + CYLINDER_RADIUS * math.sin(angle)
            points.append([x, y, z])
    
    return np.array(points)

def LineSphereIntersection(lineStart: np.ndarray, lineEnd: np.ndarray, 
                           sphereCenter: np.ndarray, sphereRadius: float) -> bool:
    """Check line-sphere intersection"""
    d = lineEnd - lineStart
    f = lineStart - sphereCenter
    
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - sphereRadius * sphereRadius
    
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return False
    
    if abs(a) < 1e-10:
        return np.linalg.norm(lineStart - sphereCenter) <= sphereRadius
    
    sqrtDiscriminant = math.sqrt(discriminant)
    t1 = (-b - sqrtDiscriminant) / (2 * a)
    t2 = (-b + sqrtDiscriminant) / (2 * a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    
    if (t1 < 0 and t2 > 1) or (t2 < 0 and t1 > 1):
        return True
    
    return False

def PointToLineDistance(point: np.ndarray, lineStart: np.ndarray, lineEnd: np.ndarray) -> float:
    """Calculate point to line distance"""
    lineVec = lineEnd - lineStart
    lineLenSq = np.dot(lineVec, lineVec)
    
    if lineLenSq < 1e-10:
        return np.linalg.norm(point - lineStart)
    
    t = np.dot(point - lineStart, lineVec) / lineLenSq
    t = max(0, min(1, t))
    
    closestPoint = lineStart + t * lineVec
    return np.linalg.norm(point - closestPoint)

def IsTargetBlockedAtTime(missilePos: np.ndarray, targetPoints: np.ndarray, 
                             cloudCenters: List[np.ndarray]) -> bool:
    """Check if target is blocked at given time"""
    if not cloudCenters:
        return False
    
    for targetPoint in targetPoints:
        pointBlocked = False
        
        for cloudCenter in cloudCenters:
            if LineSphereIntersection(missilePos, targetPoint, cloudCenter, CLOUD_RADIUS):
                pointBlocked = True
                break
        
        if not pointBlocked:
            return False
    
    return True

def CalculateBlockingCoverage(bombParams: List[Tuple[float, float, float, float]], 
                              timeStep: float = 0.01) -> Dict[str, Any]:
    """Calculate blocking coverage"""
    print(f"[Q3] Calculating coverage, time step={timeStep}s")
    
    targetPoints = GenerateCylinderSamplingPoints(nCircumference = 24, nHeight = 5)
    print(f"[Q3] Generated {len(targetPoints)} target points")
    
    explosions = []
    for i, (tDrop, fuseDelay, speed, heading) in enumerate(bombParams):
        tExplosion = tDrop + fuseDelay
        explosionPos = ExplosionPosition(tDrop, fuseDelay, speed, heading)
        explosions.append({
            'bomb_id': i + 1,
            't_drop': tDrop,
            't_explosion': tExplosion,
            'explosion_pos': explosionPos,
            'speed': speed,
            'heading': heading
        })
        print(f"[Q3] Bomb {i+1}: drop={tDrop:.3f}s, explosion={tExplosion:.3f}s, "
              f"pos=({explosionPos[0]:.1f}, {explosionPos[1]:.1f}, {explosionPos[2]:.1f})")
    
    missileHitTime = np.linalg.norm(M1_INIT - FAKE_TARGET) / MISSILE_SPEED
    print(f"[Q3] Missile hit time: {missileHitTime:.3f}s")
    
    earliestExplosion = min(exp['t_explosion'] for exp in explosions)
    latestEffectEnd = max(exp['t_explosion'] + CLOUD_EFFECT_TIME for exp in explosions)
    
    simStartTime = max(0, earliestExplosion - 1.0)
    simEndTime = min(missileHitTime, latestEffectEnd)
    
    print(f"[Q3] Simulation range: {simStartTime:.3f}s to {simEndTime:.3f}s")
    
    if simEndTime <= simStartTime:
        print("[Q3] Warning: No valid simulation time range")
        return {
            'total_blocking_time': 0.0,
            'blocking_intervals': [],
            'individual_bomb_times': [0.0] * len(bombParams),
            'simulation_details': {
                'time_range': (simStartTime, simEndTime),
                'total_time_points': 0
            }
        }
    
    timePoints = np.arange(simStartTime, simEndTime + timeStep, timeStep)
    blockingMask = np.zeros(len(timePoints), dtype = bool)
    individualBlockingMasks = [np.zeros(len(timePoints), dtype = bool) for _ in range(len(bombParams))]
    
    print(f"[Q3] Starting simulation with {len(timePoints)} time points...")
    
    for i, t in enumerate(timePoints):
        if i % 1000 == 0:
            progress = i / len(timePoints) * 100
            print(f"[Q3] Progress: {progress:.1f}%")
        
        missilePos = MissilePosition(t)
        
        activeClouds = []
        for j, explosion in enumerate(explosions):
            cloudCenter = CloudCenterPosition(
                explosion['explosion_pos'], 
                explosion['t_explosion'], 
                t
            )
            if cloudCenter is not None:
                activeClouds.append(cloudCenter)
                
                if IsTargetBlockedAtTime(missilePos, targetPoints, [cloudCenter]):
                    individualBlockingMasks[j][i] = True
        
        if IsTargetBlockedAtTime(missilePos, targetPoints, activeClouds):
            blockingMask[i] = True
    
    print("[Q3] Simulation completed, analyzing results...")
    
    totalBlockingTime = np.sum(blockingMask) * timeStep
    individualBombTimes = [np.sum(mask) * timeStep for mask in individualBlockingMasks]
    
    blockingIntervals = []
    inInterval = False
    intervalStart = None
    
    for i, blocked in enumerate(blockingMask):
        if blocked and not inInterval:
            inInterval = True
            intervalStart = timePoints[i]
        elif not blocked and inInterval:
            inInterval = False
            intervalEnd = timePoints[i-1] if i > 0 else timePoints[i]
            blockingIntervals.append((intervalStart, intervalEnd))
    
    if inInterval:
        blockingIntervals.append((intervalStart, timePoints[-1]))
    
    result = {
        'total_blocking_time': totalBlockingTime,
        'blocking_intervals': blockingIntervals,
        'individual_bomb_times': individualBombTimes,
        'simulation_details': {
            'time_range': (simStartTime, simEndTime),
            'time_step': timeStep,
            'total_time_points': len(timePoints),
            'missile_hit_time': missileHitTime,
            'target_sampling_points': len(targetPoints)
        }
    }
    
    return result

def LoadSolutionFromExcel(excelPath: str) -> Dict[str, Any]:
    """Load solution from Excel file"""
    print(f"[Q3] Loading solution from: {excelPath}")
    
    try:
        df = pd.read_excel(excelPath)
        print(f"[Q3] Successfully read Excel file with {len(df)} rows")
        
        bombs = []
        
        for idx, row in df.iterrows():
            heading = ParseHeadingFromString(row['Direction_degrees'])
            speed = float(row['Speed_m_per_s'])
            bombId = int(row['Smoke_Bomb_Number'])
            
            dropX = float(row['Drop_Point_X_m'])
            dropY = float(row['Drop_Point_Y_m'])
            dropZ = float(row['Drop_Point_Z_m'])
            
            explosionX = float(row['Detonation_Point_X_m'])
            explosionY = float(row['Detonation_Point_Y_m'])
            explosionZ = float(row['Detonation_Point_Z_m'])
            
            dropPos = np.array([dropX, dropY, dropZ])
            explosionPos = np.array([explosionX, explosionY, explosionZ])
            
            distanceToDrop = np.linalg.norm(dropPos[:2] - FY1_INIT[:2])
            tDrop = distanceToDrop / speed
            
            horizontalDrift = np.linalg.norm(explosionPos[:2] - dropPos[:2])
            fuseDelay = horizontalDrift / speed
            
            bombs.append({
                'bomb_id': bombId,
                't_drop': tDrop,
                'fuse_delay': fuseDelay,
                'speed': speed,
                'heading': heading,
                'drop_pos': dropPos,
                'explosion_pos': explosionPos,
                'individual_effect_time': float(row.get('Effective_Shielding_Duration_s', 0.0))
            })
        
        bombs.sort(key = lambda x: x['bomb_id'])
        
        solution = {
            'bombs': bombs,
            'common_heading': bombs[0]['heading'] if bombs else 0.0,
            'common_speed': bombs[0]['speed'] if bombs else 0.0,
            'bomb_params': [(b['t_drop'], b['fuse_delay'], b['speed'], b['heading']) 
                           for b in bombs]
        }
        
        print("[Q3] Solution data parsed successfully")
        for bomb in bombs:
            print(f"[Q3] Bomb {bomb['bomb_id']}: drop={bomb['t_drop']:.3f}s, "
                  f"fuse={bomb['fuse_delay']:.3f}s, speed={bomb['speed']:.1f}m/s")
        
        return solution
        
    except Exception as e:
        print(f"[Q3] Error reading Excel file: {e}")
        raise

def ParseHeadingFromString(headingStr: str) -> float:
    """Parse heading angle from string"""
    if isinstance(headingStr, (int, float)):
        return float(headingStr)
    
    import re
    match = re.search(r'([-+]?[\d\.]+)', str(headingStr))
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Cannot parse heading angle: {headingStr}")



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description = 'Q3 Checker')
    parser.add_argument('--excel', type = str, default = 'result3.xlsx',
                       help = 'Solution Excel file path')
    parser.add_argument('--time-step', type = float, default = 0.01,
                       help = 'Time step (seconds)')
    parser.add_argument('--target-points', type = int, default = 24,
                       help = 'Cylinder circumference sampling points')
    parser.add_argument('--height-layers', type = int, default = 5,
                       help = 'Cylinder height layers')
    parser.add_argument('--verbose', action = 'store_true',
                       help = 'Show detailed output')
    
    args = parser.parse_args()
    
    print("Q3Checker - High precision Q3 verifier")
    print(f"Using solution file: {args.excel}")
    print(f"Time step: {args.time_step}s")
    
    try:
        solution = LoadSolutionFromExcel(args.excel)
        
        startTime = time.time()
        results = CalculateBlockingCoverage(
            solution['bomb_params'], 
            timeStep = args.time_step
        )
        endTime = time.time()
        
        print(f"\n[Q3] Simulation completed in {endTime - startTime:.2f}s")
        
        print(f"\n[Q3] RESULTS:")
        print(f"Total blocking time: {results['total_blocking_time']:.6f}s")
        print(f"Blocking intervals: {len(results['blocking_intervals'])}")
        
        if args.verbose:
            print(f"\n[Q3] Parameters:")
            print(f"  Gravity: {G} m/s²")
            print(f"  Cloud radius: {CLOUD_RADIUS} m")
            print(f"  Sink speed: {CLOUD_SINK_SPEED} m/s")
            print(f"  Effect time: {CLOUD_EFFECT_TIME} s")
            print(f"  Missile speed: {MISSILE_SPEED} m/s")
        
        return results['total_blocking_time']
        
    except Exception as e:
        print(f"[Q3] Error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    blockingTime = main()
    print(f"\n[Q3] Final result: {blockingTime:.6f}s blocking time")
