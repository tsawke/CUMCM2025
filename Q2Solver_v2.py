import math
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time
import traceback

# Constants (from Q1Solver and problem context)
g = 9.8  # gravity acceleration (m/s^2)
MISSILE_SPEED = 300.0  # missile speed (m/s)
SMOG_R = 10.0          # smoke cloud radius (m)
SMOG_SINK_SPEED = 3.0  # smoke cloud sinking speed (m/s)
SMOG_EFFECT_TIME = 20.0  # smoke screen effective duration (s)
FY1_INIT = np.array([17800.0, 0.0, 1800.0], dtype=float)  # initial position of FY1
M1_INIT = np.array([20000.0, 0.0, 2000.0], dtype=float)   # initial position of M1 (missile)
FAKE_TARGET_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=float)  # target (sensor) at origin
EPS = 1e-12

# Parameter bounds (optimized for better convergence)
HEADING_MIN, HEADING_MAX = 0.0, math.pi         # radians [0, π) - focus on forward directions
SPEED_MIN, SPEED_MAX = 100.0, 250.0            # m/s - reasonable speed range
DROP_MIN, DROP_MAX = 5.0, 25.0                 # s - optimal timing window
FUSE_MIN, FUSE_MAX = 1.0, 8.0                  # s - reasonable fuse delay

# Helper functions (from Q1Solver)
def Unit(v):
    """Return unit vector in direction of v (or zero vector if v is near zero)."""
    n = np.linalg.norm(v)
    return v if n < EPS else (v / n)

def MissileState(t, mInit):
    """Return position and velocity of missile M1 at time t."""
    dirToOrigin = Unit(FAKE_TARGET_ORIGIN - mInit)
    v = MISSILE_SPEED * dirToOrigin
    return mInit + v * t, v

def UavStateHorizontal(t, uavInit, uavSpeed, headingRad):
    """Return position and velocity of UAV (FY1) at time t, moving horizontally at given heading and speed."""
    vx = uavSpeed * math.cos(headingRad)
    vy = uavSpeed * math.sin(headingRad)
    # Assume constant altitude (horizontal flight), so z remains the same
    return np.array([uavInit[0] + vx * t, uavInit[1] + vy * t, uavInit[2]], dtype=float), \
           np.array([vx, vy, 0.0], dtype=float)

def PreCalCylinderPoints(nPhi, nZ, dtype=np.float64):
    """Pre-calculate sample points on the cylindrical smoke cloud (surface points)."""
    b = np.array([0.0, 200.0, 0.0], dtype=dtype)  # base center of cylinder (from Q1Solver)
    r, h = dtype(7.0), dtype(10.0)                # cylinder radius and height (CYLINDER_R, CYLINDER_H from Q1Solver)
    # Circumferential angle samples
    phis = np.linspace(0.0, 2.0 * math.pi, nPhi, endpoint=False, dtype=dtype)
    c, s = np.cos(phis), np.sin(phis)
    # Base ring points
    ring = np.stack([r * c, r * s, np.zeros_like(c)], axis=1).astype(dtype)
    pts = [b + ring, b + np.array([0.0, 0.0, h], dtype=dtype) + ring]
    if nZ >= 2:
        # Sample intermediate layers between base and top
        for z in np.linspace(0.0, h, nZ, dtype=dtype):
            pts.append(b + np.array([0.0, 0.0, z], dtype=dtype) + ring)
    p = np.vstack(pts).astype(dtype)
    return p

def ConeAllPointsIn(m, c, p, rCloud=SMOG_R, margin=EPS, block=8192):
    """
    Check strict cone criterion: returns True if *all* points in array p 
    (representing the cross-section) lie inside the occlusion cone of the smoke cloud.
    m: missile position (apex of viewing cone), c: center of smoke cloud, p: points to test.
    """
    v = c - m
    l = np.linalg.norm(v)
    if l <= EPS or rCloud >= l:
        # If missile and cloud center are extremely close or cloud radius covers distance, occlusion is trivial.
        return True
    # cos(alpha) = sqrt(1 - (rCloud / l)^2) for cone half-angle
    cosAlpha = math.sqrt(max(0.0, 1.0 - (rCloud / l) ** 2))
    # Check all points in blocks
    for i in range(0, len(p), block):
        w = p[i : i + block] - m        # vectors from missile to each point in block
        wn = np.linalg.norm(w, axis=1) + EPS
        lhs = w @ v                     # dot product with v
        rhs = wn * l * cosAlpha
        if not np.all(lhs + margin >= rhs):
            return False  # if any point lies outside the cone, criterion fails
    return True

def EvaluateCombination(heading, speed, drop_time, fuse_delay, dt=0.002, nphi=240, nz=7):
    """
    Compute occlusion duration for a given combination of parameters.
    Returns: (occlusion_time, explosion_point, missile_hit_time)
    """
    # Calculate explosion (smoke release) point based on UAV trajectory and fuse delay
    dropPos, uavV = UavStateHorizontal(drop_time, FY1_INIT, speed, heading)
    expl_xy = dropPos[:2] + uavV[:2] * fuse_delay        # horizontal displacement during fuse delay
    expl_z = dropPos[2] - 0.5 * g * (fuse_delay ** 2)    # vertical drop due to gravity
    expl_pos = np.array([expl_xy[0], expl_xy[1], expl_z], dtype=float)
    # If explosion is at or below ground level, smoke won't be effective
    if expl_pos[2] <= 0:
        # Missile hit time (approximate) – distance from M1 start to origin / speed
        hit_time = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
        return 0.0, expl_pos, hit_time
    # Calculate missile hit time (constant for given scenario)
    hit_time = float(np.linalg.norm(M1_INIT - FAKE_TARGET_ORIGIN) / MISSILE_SPEED)
    # Time window from smoke explosion until it dissipates or until missile hits
    explode_time = drop_time + fuse_delay
    t0 = explode_time
    t1 = min(explode_time + SMOG_EFFECT_TIME, hit_time)
    if t0 >= t1:
        # If explosion happens after missile impact, no occlusion time
        return 0.0, expl_pos, hit_time
    # Generate time grid for evaluation
    t_grid = np.arange(t0, t1 + EPS, dt, dtype=float)
    pts = PreCalCylinderPoints(nphi, nz, dtype=float)  # sample points on smoke cloud cross-section
    mask = np.zeros(len(t_grid), dtype=bool)
    # Check occlusion at each time step
    for j, t in enumerate(t_grid):
        m_pos, _ = MissileState(float(t), M1_INIT)  # missile position at time t
        # Cloud center at time t (smoke sinks over time after explosion)
        cloud_center = np.array([
            expl_pos[0],
            expl_pos[1],
            expl_pos[2] - SMOG_SINK_SPEED * max(0.0, float(t) - t0)
        ], dtype=float)
        # Strict cone check: is missile fully within smoke cover from viewpoint?
        mask[j] = ConeAllPointsIn(m_pos, cloud_center, pts, rCloud=SMOG_R, margin=EPS)
    occlusion_duration = float(np.count_nonzero(mask) * dt)
    return occlusion_duration, expl_pos, hit_time

def FindOptimalParameters(pop_size=30, iterations=50, workers=None):
    """
    Perform PSO to find the optimal parameters maximizing occlusion time.
    Returns: (best_occlusion, best_params, explosion_point, hit_time)
    """
    start_time = time.time()
    print(f"[初始化] 创建{len(particles) if 'particles' in locals() else pop_size}个粒子...")

    # Initialize swarm with random positions within bounds and zero initial velocity
    particles = []
    for i in range(pop_size):
        # Random initial position within bounds
        init_heading = np.random.uniform(HEADING_MIN, HEADING_MAX)
        init_speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
        init_drop = np.random.uniform(DROP_MIN, DROP_MAX)
        init_fuse = np.random.uniform(FUSE_MIN, FUSE_MAX)
        # Small random initial velocity for each dimension
        init_vel = np.array([
            np.random.uniform(-1, 1) * (HEADING_MAX - HEADING_MIN) * 0.1,   # heading velocity (rad/s)
            np.random.uniform(-1, 1) * (SPEED_MAX - SPEED_MIN) * 0.1,     # speed velocity
            np.random.uniform(-1, 1) * (DROP_MAX - DROP_MIN) * 0.1,       # drop_time velocity
            np.random.uniform(-1, 1) * (FUSE_MAX - FUSE_MIN) * 0.1        # fuse_delay velocity
        ], dtype=float)
        particles.append({
            "pos": np.array([init_heading, init_speed, init_drop, init_fuse], dtype=float),
            "vel": init_vel,
            "pbest_pos": None,
            "pbest_val": 0.0
        })

    global_best_val = 0.0
    global_best_pos = None
    global_best_expl = None
    global_hit_time = 0.0  # will store missile hit time (constant)
    # PSO hyperparameters
    w = 0.5    # inertia weight
    c1 = 1.5   # cognitive (particle) weight
    c2 = 1.5   # social (swarm) weight

    # Determine number of workers for parallel evaluation
    if workers is None:
        workers = os.cpu_count() or 1

    print(f"[参数] 粒子数: {pop_size}, 迭代次数: {iterations}, 并行线程: {workers}")
    print(f"[开始] 开始粒子群优化...")

    # Optimization loop
    for it in range(iterations):
        iteration_start = time.time()

        # Evaluate all particles in parallel for current positions
        results = [None] * pop_size
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(EvaluateCombination, p["pos"][0], p["pos"][1], p["pos"][2], p["pos"][3]): idx
                           for idx, p in enumerate(particles)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        occlusion, expl_point, hit_time = fut.result()
                        results[idx] = (occlusion, expl_point, hit_time)
                    except Exception as e:
                        print(f"[警告] 粒子{idx}评估失败: {e}")
                        results[idx] = (0.0, np.zeros(3), 0.0)  # 默认值
        except Exception as e:
            print(f"[错误] 并行评估失败: {e}")
            traceback.print_exc()
            continue

        # Update personal and global bests
        for idx, (occlusion, expl_point, hit_time) in enumerate(results):
            p = particles[idx]
            # Initialize pbest if first iteration
            if p["pbest_pos"] is None:
                p["pbest_pos"] = p["pos"].copy()
                p["pbest_val"] = occlusion
            # Personal best update
            if occlusion > p["pbest_val"]:
                p["pbest_val"] = occlusion
                p["pbest_pos"] = p["pos"].copy()
            # Global best update
            if occlusion > global_best_val:
                global_best_val = occlusion
                global_best_pos = p["pos"].copy()
                global_best_expl = expl_point.copy()
                global_hit_time = hit_time

        iteration_time = time.time() - iteration_start
        progress = (it + 1) / iterations * 100

        # 每5%显示进度
        if int(progress) % 5 == 0 or it == iterations - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (it + 1) * (iterations - it - 1)
            print(f"   [进度] 第{it+1:2d}/{iterations}代 ({progress:5.1f}%) | "
                  f"用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s | "
                  f"最佳遮蔽时长: {global_best_val:.3f}s")
        # Update particle velocities and positions for next iteration
        for p in particles:
            if global_best_pos is None:
                continue  # not updated yet (should not happen after first iteration)
            # random coefficients
            r1, r2 = np.random.rand(2)
            # velocity update (inertia + cognitive + social components)
            p["vel"][0] = w * p["vel"][0] + c1 * r1 * (p["pbest_pos"][0] - p["pos"][0]) + c2 * r2 * (global_best_pos[0] - p["pos"][0])
            p["vel"][1] = w * p["vel"][1] + c1 * r1 * (p["pbest_pos"][1] - p["pos"][1]) + c2 * r2 * (global_best_pos[1] - p["pos"][1])
            p["vel"][2] = w * p["vel"][2] + c1 * r1 * (p["pbest_pos"][2] - p["pos"][2]) + c2 * r2 * (global_best_pos[2] - p["pos"][2])
            p["vel"][3] = w * p["vel"][3] + c1 * r1 * (p["pbest_pos"][3] - p["pos"][3]) + c2 * r2 * (global_best_pos[3] - p["pos"][3])
            # Update position
            p["pos"] += p["vel"]
            # Apply boundary conditions (clamp positions and reflect velocity if out of bounds)
            # Heading: wrap around 0 to 2π
            if p["pos"][0] < HEADING_MIN or p["pos"][0] >= HEADING_MAX:
                # Wrap angle into [0, 2π)
                p["pos"][0] = p["pos"][0] % (2.0 * math.pi)
                # No need to reflect velocity for angle due to wrap-around
            # Speed, Drop, Fuse: clamp within [min, max]
            if p["pos"][1] < SPEED_MIN:
                p["pos"][1] = SPEED_MIN
                p["vel"][1] *= -0.5  # bounce back with damping
            elif p["pos"][1] > SPEED_MAX:
                p["pos"][1] = SPEED_MAX
                p["vel"][1] *= -0.5
            if p["pos"][2] < DROP_MIN:
                p["pos"][2] = DROP_MIN
                p["vel"][2] *= -0.5
            elif p["pos"][2] > DROP_MAX:
                p["pos"][2] = DROP_MAX
                p["vel"][2] *= -0.5
            if p["pos"][3] < FUSE_MIN:
                p["pos"][3] = FUSE_MIN
                p["vel"][3] *= -0.5
            elif p["pos"][3] > FUSE_MAX:
                p["pos"][3] = FUSE_MAX
                p["vel"][3] *= -0.5

    total_time = time.time() - start_time
    print(f"\n[完成] 优化完成! 总用时: {total_time:.2f}s")
    print(f"[结果] 最佳遮蔽时长: {global_best_val:.6f}s")

    # Return the best found solution
    return global_best_val, global_best_pos, global_best_expl, global_hit_time

def main():
    ap = argparse.ArgumentParser(description="Q2问题：优化烟幕屏障参数以最大化M1导弹遮蔽时长")
    ap.add_argument("--pop_size", type=int, default=8, help="PSO种群大小（粒子数量）")
    ap.add_argument("--iterations", type=int, default=15, help="PSO迭代次数")
    ap.add_argument("--workers", type=int, default=None, help="并行评估线程数（默认：使用全部CPU核心）")
    args = ap.parse_args()

    # 参数验证
    if args.pop_size < 3:
        print("[警告] 粒子数太少，建议至少3个，已调整为3")
        args.pop_size = 3
    if args.iterations < 5:
        print("[警告] 迭代次数太少，建议至少5次，已调整为5")
        args.iterations = 5

    print("=" * 80)
    print("🚀 Q2问题求解：最大化M1导弹遮蔽时长优化")
    print("=" * 80)
    print(f"📊 粒子数: {args.pop_size}, 迭代次数: {args.iterations}, 并行线程: {args.workers or '自动'}")
    print("=" * 80)
    # Run the optimization
    best_val, best_pos, best_expl, hit_time = FindOptimalParameters(pop_size=args.pop_size, iterations=args.iterations, workers=args.workers)

    # 检查优化结果
    if best_pos is None or best_val == 0.0:
        print("\n⚠️ 警告：未找到有效的遮蔽方案")
        print("可能原因：")
        print("  • 参数范围设置不合理")
        print("  • 初始随机种子导致局部最优")
        print("建议：")
        print("  • 增加粒子数和迭代次数")
        print("  • 重新运行程序（随机种子不同）")
        print("  • 检查参数范围设置")
        return

    # Extract best parameters
    best_heading, best_speed, best_drop, best_fuse = best_pos
    # Save results to a text file (format consistent with Q1Results.txt)
    with open("Q2Results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Q2 Calculation Results Report\n")
        f.write("=" * 80 + "\n\n")
        f.write("Calculation Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Heading angle range = [0, 6.283185] rad (0° to 360°)\n")
        f.write(f"Flight speed range = [{SPEED_MIN:.1f}, {SPEED_MAX:.1f}] m/s\n")
        f.write(f"Drop time range = [{DROP_MIN:.1f}, {DROP_MAX:.1f}] s\n")
        f.write(f"Fuse delay range = [{FUSE_MIN:.1f}, {FUSE_MAX:.1f}] s\n")
        f.write(f"Optimization algorithm = PSO (swarm size = {args.pop_size}, iterations = {args.iterations})\n")
        f.write(f"Parallel evaluation threads = {args.workers or os.cpu_count() or 1}\n\n")
        f.write("Calculation Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Occlusion duration = {best_val:.6f} seconds\n")
        f.write(f"Optimal heading angle = {best_heading:.6f} rad\n")
        f.write(f"Optimal flight speed = {best_speed:.6f} m/s\n")
        f.write(f"Optimal drop time = {best_drop:.6f} s\n")
        f.write(f"Optimal fuse delay = {best_fuse:.6f} s\n")
        f.write(f"Explosion point = ({best_expl[0]:.6f}, {best_expl[1]:.6f}, {best_expl[2]:.6f})\n")
        f.write(f"Missile hit time ≈ {hit_time:.6f} seconds\n\n")
        f.write("=" * 80 + "\n")
    # Console output of results
    print("\n" + "=" * 80)
    print("🎯 优化结果")
    print("=" * 80)
    print(f"⏱️  最大遮蔽时长: {best_val:.6f} 秒")
    print(f"🎯 最优航向角: {best_heading:.6f} 弧度 ({np.degrees(best_heading):.1f}°)")
    print(f"💨 最优飞行速度: {best_speed:.6f} m/s")
    print(f"📦 最优投放时间: {best_drop:.6f} 秒")
    print(f"⏰ 最优引信延时: {best_fuse:.6f} 秒")
    print(f"💥 爆炸点坐标: ({best_expl[0]:.2f}, {best_expl[1]:.2f}, {best_expl[2]:.2f})")
    print(f"🚀 导弹命中时间: {hit_time:.6f} 秒")
    print(f"💾 结果已保存至: Q2Results.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
