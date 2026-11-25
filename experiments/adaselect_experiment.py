







from __future__ import annotations
import argparse, csv, hashlib, json, math, multiprocessing as mp, os, platform, random
import socket, statistics, struct, subprocess, sys, time
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
BASE_PORT = 31000
MAX_WORKERS = 20
TASKS_PER_RUN = 90
H_MS = 4.0
QREF_MS = 4.0
V_MS_PER_STEP = 0.007
EXTRA_IMPAIRMENT = 1.5
BASE_SALT = b"adaselect-base-v3"
DUMMY_SALT = b"adaselect-dummy-v3"
POLICY_ORDER = ["Full", "Class-Top-K", "RR", "Bid-only", "Obs.+Top-K", "AdaSelect"]


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionError("socket closed before complete frame")
        buf.extend(part)
    return bytes(buf)


def recv_json(conn: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", recv_exact(conn, 4))[0]
    return json.loads(recv_exact(conn, size))


def send_json(conn: socket.socket, obj: dict[str, Any]) -> int:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)
    return 4 + len(payload)


def do_pbkdf2(data: bytes, iters: int, role: int, extra: float) -> tuple[float, str]:
    
    t0 = time.perf_counter_ns()
    digest = hashlib.pbkdf2_hmac("sha256", data, BASE_SALT, int(iters), dklen=32)
    dummy_iters = int(round(iters * ((role - 1) + extra)))
    if dummy_iters > 0:
        hashlib.pbkdf2_hmac("sha256", data, DUMMY_SALT, dummy_iters, dklen=32)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    return elapsed_ms, digest.hex()


def worker_server(worker_id: int, roles: list[int], stop_event, ready_event, port: int) -> None:
    
    unit_ms: list[float] = []
    for cls in range(3):
        samples = []
        data = hashlib.sha256(f"local-cal-{worker_id}-{cls}".encode()).digest()
        for rep in range(2):
            elapsed, _ = do_pbkdf2(data, 1000, roles[cls], 0.0)
            samples.append(elapsed / 1000.0)
        unit_ms.append(statistics.median(samples))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, port))
    server.listen(32)
    server.settimeout(0.2)
    ready_event.set()
    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(None)
                while not stop_event.is_set():
                    try:
                        req = recv_json(conn)
                    except (ConnectionError, OSError):
                        break
                    try:
                        op = req["op"]
                        cls = int(req["cls"])
                        iters = int(req["iters"])
                        if op == "bid":
                            send_json(conn, {"estimated_ms": iters * unit_ms[cls]})
                        elif op in ("probe", "execute"):
                            extra = float(req.get("extra", 0.0)) if req.get("impaired", False) else 0.0
                            elapsed, digest = do_pbkdf2(bytes.fromhex(req["data"]), iters, roles[cls], extra)
                            send_json(conn, {"ms": elapsed, "digest": digest})
                        else:
                            send_json(conn, {"error": f"unknown operation: {op}"})
                    except Exception as exc:
                        try:
                            send_json(conn, {"error": repr(exc)})
                        except Exception:
                            break
    finally:
        server.close()


class WorkerCluster:
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.ctx = mp.get_context("fork")
        self.stop_event = self.ctx.Event()
        self.processes = []
        self.ready_events = []
        self.max_workers = max_workers
        for i in range(max_workers):
            roles = [1 if i % 3 == c else 2 for c in range(3)]
            ready = self.ctx.Event()
            proc = self.ctx.Process(
                target=worker_server,
                args=(i, roles, self.stop_event, ready, BASE_PORT + i),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
            self.ready_events.append(ready)
        for i, ev in enumerate(self.ready_events):
            if not ev.wait(15):
                raise RuntimeError(f"worker {i} failed to become ready")
        self.connections = []
        for i in range(max_workers):
            conn = socket.create_connection((HOST, BASE_PORT + i), timeout=10)
            conn.settimeout(10)
            self.connections.append(conn)

    def call(self, worker_id: int, request: dict[str, Any], rpc_delay_ms: float = 0.0):
        if rpc_delay_ms > 0:
            time.sleep(rpc_delay_ms / 1000.0)
        req_payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        t0 = time.perf_counter_ns()
        conn = self.connections[worker_id]
        conn.sendall(struct.pack("!I", len(req_payload)) + req_payload)
        resp_size = struct.unpack("!I", recv_exact(conn, 4))[0]
        resp_payload = recv_exact(conn, resp_size)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        response = json.loads(resp_payload)
        if "error" in response:
            raise RuntimeError(response["error"])
        byte_count = 8 + len(req_payload) + len(resp_payload)
        return response, byte_count, elapsed_ms

    def close(self):
        for conn in getattr(self, "connections", []):
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self.stop_event.set()
        for p in self.processes:
            p.join(timeout=2)
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)


def coordinator_calibration(cluster: WorkerCluster, n: int, seed: int) -> list[list[float]]:
    theta0 = [[0.0] * 3 for _ in range(n)]
    for i in range(n):
        for cls in range(3):
            samples = []
            for rep in range(2):
                data = hashlib.sha256(f"coord-cal-{seed}-{i}-{cls}-{rep}".encode()).digest()
                req = {"op": "probe", "cls": cls, "iters": 1000, "data": data.hex(), "impaired": False}
                response, _, _ = cluster.call(i, req)
                samples.append(response["ms"] / 1000.0)
            theta0[i][cls] = statistics.median(samples)
    return theta0


def make_tasks(seed: int, task_mode: str = "cyclic", count: int = TASKS_PER_RUN):
    rng = random.Random(seed * 17 + (0 if task_mode == "cyclic" else 911))
    tasks = []
    for j in range(count):
        cls = j % 3 if task_mode == "cyclic" else rng.randrange(3)
        iters = rng.randint(900, 1800)
        data = hashlib.sha256(f"task-{seed}-{task_mode}-{j}".encode()).digest()
        expected = hashlib.pbkdf2_hmac("sha256", data, BASE_SALT, iters, dklen=32).hex()
        tasks.append({"cls": cls, "iters": iters, "data": data, "expected": expected})
    return tasks


def make_async_windows(seed: int, n: int):
    rng = random.Random(seed * 31 + 1777)
    windows = [[(0.0, 0.0) for _ in range(3)] for _ in range(n)]
    for i in range(n):
        for cls in range(3):
            if i % 3 == cls:
                start = rng.uniform(0.15, 0.30)
                end = rng.uniform(0.45, 0.65)
                windows[i][cls] = (start, end)
    return windows


def impaired_workers(n: int, cls: int, phase: float, setting: str, windows=None):
    flags = [False] * n
    if setting == "recovery":
        for i in range(n):
            flags[i] = (i % 3 == cls and 0.25 <= phase < 0.55)
    elif setting == "async":
        for i in range(n):
            if i % 3 == cls:
                start, end = windows[i][cls]
                flags[i] = start <= phase < end
    return flags


def estimate_state(theta, theta0, last, cls, iters):
    pred = [theta[i][cls] * iters for i in range(len(theta))]
    score = [1.0 - pred[i] / H_MS for i in range(len(theta))]
    return pred, score


def eligible_candidates(theta, theta0, last, cls, iters, step):
    pred, _ = estimate_state(theta, theta0, last, cls, iters)
    incumbent = min(range(len(theta)), key=lambda i: pred[i])
    age_c = step - last[incumbent][cls]
    upper_c = 1.12 * pred[incumbent] + V_MS_PER_STEP * age_c
    eligible = []
    for i in range(len(theta)):
        if i == incumbent:
            continue
        age_i = step - last[i][cls]
        if age_i < 4:
            continue
        ref_i = iters * theta0[i][cls]
        if pred[i] <= 1.50 * ref_i:
            continue
        if ref_i >= 0.90 * pred[incumbent]:
            continue
        lower_i = max(0.0, 0.88 * pred[i] - V_MS_PER_STEP * age_i)
        if upper_c - lower_i <= 0.020:
            continue
        gain = pred[incumbent] - ref_i + upper_c - lower_i
        deadline_factor = 1.0 + pred[i] / CURRENT_DEADLINE_MS
        age_factor = min(age_i, 24)
        eligible.append((i, gain, age_factor, deadline_factor))
    return incumbent, eligible


CURRENT_DEADLINE_MS = 0.45


def select_diagnostic(policy: str, theta, theta0, last, cls, iters, step, n, rng, rr_orders, rr_ptr):
    if policy == "RR":
        idx = rr_orders[cls][rr_ptr[cls] % n]
        rr_ptr[cls] += 1
        return idx
    if policy not in ("AdaSelect", "Obs.+Top-K", "Rand-Obs", "No-age", "No-deadline", "No-gain"):
        return None
    _, eligible = eligible_candidates(theta, theta0, last, cls, iters, step)
    if not eligible:
        return None
    if policy == "Rand-Obs":
        return rng.choice([item[0] for item in eligible])
    def priority(item):
        _, gain, age_factor, deadline_factor = item
        if policy == "No-age":
            return gain / deadline_factor
        if policy == "No-deadline":
            return gain * age_factor
        if policy == "No-gain":
            return age_factor / deadline_factor
        return gain * age_factor / deadline_factor
    return max(eligible, key=priority)[0]


def run_trial(cluster: WorkerCluster, n: int, seed: int, policy: str, tasks, theta0,
              setting: str, deadline_ms: float = 0.45, eta: float = 0.07, eps: float = 0.035,
              task_mode: str = "cyclic", rpc_delay_ms: float = 0.0, async_windows=None,
              experiment: str = "main"):
    global CURRENT_DEADLINE_MS
    CURRENT_DEADLINE_MS = deadline_ms
    theta = [row[:] for row in theta0]
    last = [[0] * 3 for _ in range(n)]
    rng = random.Random(seed * 100003 + sum(ord(c) for c in policy) + int(deadline_ms * 1000))
    rr_orders = []
    for cls in range(3):
        order = list(range(n))
        random.Random(seed * 1009 + cls * 97 + n).shuffle(order)
        rr_orders.append(order)
    rr_ptr = [0, 0, 0]

    total_success = 0
    post_success = 0
    post_tasks = 0
    messages = 0
    bytes_total = 0
    diagnoses = 0
    bid_rpcs = 0
    cycle_times = []

    for j, task in enumerate(tasks):
        cls, iters, data, expected = task["cls"], task["iters"], task["data"], task["expected"]
        phase = j / len(tasks)
        impaired = impaired_workers(n, cls, phase, setting, async_windows)
        cycle_start = time.perf_counter_ns()

        
        cand = select_diagnostic(policy, theta, theta0, last, cls, iters, j, n, rng, rr_orders, rr_ptr)
        if cand is not None:
            req = {"op": "probe", "cls": cls, "iters": iters, "data": data.hex(),
                   "impaired": impaired[cand], "extra": EXTRA_IMPAIRMENT}
            response, nbytes, _ = cluster.call(cand, req, rpc_delay_ms)
            messages += 2; bytes_total += nbytes; diagnoses += 1
            theta[cand][cls] = 0.35 * theta[cand][cls] + 0.65 * (response["ms"] / iters)
            last[cand][cls] = j

        pred, scores = estimate_state(theta, theta0, last, cls, iters)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)

        adaptive = policy in ("Bid-only", "AdaSelect", "Rand-Obs", "No-age", "No-deadline", "No-gain")
        if policy == "Full":
            invite = order[:]
        elif policy in ("Class-Top-K", "RR", "Obs.+Top-K"):
            invite = order[:min(3, n)]
        else:
            invite = []

        bids = {}
        if adaptive:
            for i in order:
                req = {"op": "bid", "cls": cls, "iters": iters}
                response, nbytes, _ = cluster.call(i, req, rpc_delay_ms)
                messages += 2; bytes_total += nbytes; bid_rpcs += 1
                q_ms = response["estimated_ms"]
                bids[i] = scores[i] - eta * (q_ms / QREF_MS)
                invite.append(i)
                best_bid = max(bids.values())
                remaining = order[len(invite):]
                if len(invite) >= min(2, n) and (
                    not remaining or max(scores[k] for k in remaining) <= best_bid + eps
                ):
                    break
        else:
            for i in invite:
                req = {"op": "bid", "cls": cls, "iters": iters}
                response, nbytes, _ = cluster.call(i, req, rpc_delay_ms)
                messages += 2; bytes_total += nbytes; bid_rpcs += 1
                q_ms = response["estimated_ms"]
                bids[i] = scores[i] - eta * (q_ms / QREF_MS)

        chosen = max(invite, key=lambda i: bids[i])
        req = {"op": "execute", "cls": cls, "iters": iters, "data": data.hex(),
               "impaired": impaired[chosen], "extra": EXTRA_IMPAIRMENT}
        response, nbytes, _ = cluster.call(chosen, req, rpc_delay_ms)
        messages += 2; bytes_total += nbytes
        valid = response["digest"] == expected and response["ms"] <= deadline_ms
        total_success += int(valid)
        post_flag = phase >= (0.65 if setting == "async" else 0.55)
        if setting != "stable" and post_flag:
            post_tasks += 1
            post_success += int(valid)
        theta[chosen][cls] = 0.35 * theta[chosen][cls] + 0.65 * (response["ms"] / iters)
        last[chosen][cls] = j
        cycle_times.append((time.perf_counter_ns() - cycle_start) / 1e6)

    ordered = sorted(cycle_times)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "experiment": experiment, "scenario": setting, "seed": seed, "n": n,
        "policy": policy, "deadline_ms": deadline_ms, "eta": eta, "eps": eps,
        "task_mode": task_mode, "rpc_delay_ms": rpc_delay_ms, "tasks": len(tasks),
        "tcr": 100.0 * total_success / len(tasks),
        "post_tcr": 100.0 * post_success / post_tasks if post_tasks else float("nan"),
        "post_tasks": post_tasks, "messages_per_task": messages / len(tasks),
        "bytes_per_task": bytes_total / len(tasks), "diagnoses_per_task": diagnoses / len(tasks),
        "bid_rpcs_per_task": bid_rpcs / len(tasks), "p50_cycle_ms": p50,
        "p95_cycle_ms": p95, "mean_cycle_ms": statistics.mean(cycle_times),
        "calibration_rpcs": 6 * n,
    }


def env_metadata(script_path: Path):
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    try:
        openssl = subprocess.check_output([sys.executable, "-c", "import ssl; print(ssl.OPENSSL_VERSION)"], text=True).strip()
    except Exception:
        openssl = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus_visible": os.cpu_count(),
        "openssl": openssl,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "pbkdf2": "hashlib.pbkdf2_hmac('sha256', ...), dklen=32",
        "transport": "length-prefixed JSON over loopback TCP; one request and one reply per RPC",
    }


def run_suite(out_dir: Path, quick: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    rows = []
    raw_path = out_dir / "adaselect_all_runs.csv"
    fieldnames = ["experiment", "scenario", "seed", "n", "policy", "deadline_ms", "eta", "eps",
                  "task_mode", "rpc_delay_ms", "tasks", "tcr", "post_tcr", "post_tasks",
                  "messages_per_task", "bytes_per_task", "diagnoses_per_task", "bid_rpcs_per_task",
                  "p50_cycle_ms", "p95_cycle_ms", "mean_cycle_ms", "calibration_rpcs"]
    resume = getattr(run_suite, "resume", False)
    existing = []
    if resume and raw_path.exists():
        with raw_path.open(newline="") as f:
            existing = list(csv.DictReader(f))
        rows.extend(existing)
    def key(row):
        return tuple(str(row.get(k, "")) for k in ("experiment", "scenario", "seed", "n", "policy", "deadline_ms", "eta", "eps", "task_mode", "rpc_delay_ms"))
    done = {key(r) for r in existing}
    file_mode = "a" if existing else "w"
    raw_file = raw_path.open(file_mode, newline="")
    writer = csv.DictWriter(raw_file, fieldnames=fieldnames)
    if not existing:
        writer.writeheader(); raw_file.flush()
    def record(row):
        if key(row) in done:
            return
        rows.append(row)
        done.add(key(row))
        writer.writerow(row)
        raw_file.flush()
    def should_skip(experiment, scenario, seed, n, policy, deadline_ms=0.45, eta=0.07, eps=0.035, task_mode="cyclic", rpc_delay_ms=0.0):
        probe = {"experiment": experiment, "scenario": scenario, "seed": seed, "n": n, "policy": policy,
                 "deadline_ms": deadline_ms, "eta": eta, "eps": eps, "task_mode": task_mode,
                 "rpc_delay_ms": rpc_delay_ms}
        return key(probe) in done
    cluster = WorkerCluster(MAX_WORKERS)
    try:
        if quick:
            seeds_main = [1000]
            seeds_param = [2000]
            seeds_deadline = [3000]
            seeds_extra = [4000]
            ns_main = [12]
            main_policies = ["Full", "Class-Top-K", "RR", "Bid-only", "Obs.+Top-K", "AdaSelect"]
        else:
            seeds_main = list(range(1000, 1010))
            seeds_param = list(range(2000, 2010))
            seeds_deadline = list(range(3000, 3010))
            seeds_extra = list(range(4000, 4010))
            ns_main = [8, 12, 20]
            main_policies = POLICY_ORDER[:]

        
        for n in ns_main:
            for seed in seeds_main:
                theta0 = coordinator_calibration(cluster, n, seed)
                for setting in ("stable", "recovery"):
                    tasks = make_tasks(seed, "cyclic")
                    policies = main_policies[:]
                    random.Random(seed * 13 + n + (0 if setting == "stable" else 1)).shuffle(policies)
                    for policy in policies:
                        if should_skip("main", setting, seed, n, policy):
                            continue
                        row = run_trial(cluster, n, seed, policy, tasks, theta0, setting,
                                        experiment="main")
                        record(row)
                        print(f"RUN {len(rows):03d} main N={n} seed={seed} {setting} {policy} TCR={row['tcr']:.2f} msgs={row['messages_per_task']:.2f}", flush=True)

        
        if not quick:
            for seed in seeds_param:
                n = 12; theta0 = coordinator_calibration(cluster, n, seed)
                tasks = make_tasks(seed, "cyclic")
                combos = [(eta, eps) for eta in (0.03, 0.07, 0.11) for eps in (0.020, 0.035, 0.050)]
                random.Random(seed + 87).shuffle(combos)
                for eta, eps in combos:
                    if should_skip("parameter", "recovery", seed, n, "AdaSelect", eta=eta, eps=eps):
                        continue
                    row = run_trial(cluster, n, seed, "AdaSelect", tasks, theta0, "recovery",
                                    eta=eta, eps=eps, experiment="parameter")
                    record(row)
                    print(f"RUN {len(rows):03d} parameter seed={seed} eta={eta} eps={eps} TCR={row['tcr']:.2f}", flush=True)

            
            for seed in seeds_deadline:
                n = 12; theta0 = coordinator_calibration(cluster, n, seed)
                tasks = make_tasks(seed, "cyclic")
                cases = [(d, p) for d in (0.35, 0.45, 0.55) for p in ("Class-Top-K", "RR", "AdaSelect")]
                random.Random(seed + 121).shuffle(cases)
                for deadline, policy in cases:
                    if should_skip("deadline", "recovery", seed, n, policy, deadline_ms=deadline):
                        continue
                    row = run_trial(cluster, n, seed, policy, tasks, theta0, "recovery",
                                    deadline_ms=deadline, experiment="deadline")
                    record(row)
                    print(f"RUN {len(rows):03d} deadline seed={seed} d={deadline} {policy} TCR={row['tcr']:.2f}", flush=True)

            
            for seed in seeds_extra:
                n = 12; theta0 = coordinator_calibration(cluster, n, seed)
                
                tasks = make_tasks(seed, "cyclic")
                priority_policies = ["AdaSelect", "No-age", "No-deadline", "No-gain", "Rand-Obs"]
                for policy in priority_policies:
                    if should_skip("priority", "recovery", seed, n, policy):
                        continue
                    row = run_trial(cluster, n, seed, policy, tasks, theta0, "recovery",
                                    experiment="priority")
                    record(row)
                    print(f"RUN {len(rows):03d} priority seed={seed} {policy} TCR={row['tcr']:.2f}", flush=True)
                
                tasks_random = make_tasks(seed, "random")
                for policy in ("Class-Top-K", "Rand-Obs", "Obs.+Top-K", "AdaSelect"):
                    if should_skip("random_classes", "recovery", seed, n, policy, task_mode="random"):
                        continue
                    row = run_trial(cluster, n, seed, policy, tasks_random, theta0, "recovery",
                                    task_mode="random", experiment="random_classes")
                    record(row)
                    print(f"RUN {len(rows):03d} random_classes seed={seed} {policy} TCR={row['tcr']:.2f}", flush=True)
                
                async_windows = make_async_windows(seed, n)
                for policy in ("Class-Top-K", "Rand-Obs", "Obs.+Top-K", "AdaSelect"):
                    if should_skip("async_recovery", "async", seed, n, policy):
                        continue
                    row = run_trial(cluster, n, seed, policy, tasks, theta0, "async",
                                    async_windows=async_windows, experiment="async_recovery")
                    record(row)
                    print(f"RUN {len(rows):03d} async_recovery seed={seed} {policy} TCR={row['tcr']:.2f}", flush=True)
                
                for policy in ("Class-Top-K", "Rand-Obs", "Obs.+Top-K", "AdaSelect"):
                    if should_skip("rpc_delay", "recovery", seed, n, policy, rpc_delay_ms=0.5):
                        continue
                    row = run_trial(cluster, n, seed, policy, tasks, theta0, "recovery",
                                    rpc_delay_ms=0.5, experiment="rpc_delay")
                    record(row)
                    print(f"RUN {len(rows):03d} rpc_delay seed={seed} {policy} TCR={row['tcr']:.2f}", flush=True)

        metadata = env_metadata(script_path)
        metadata.update({
            "run_count": len(rows),
            "task_execution_count": sum(int(r["tasks"]) for r in rows),
            "tasks_per_run": TASKS_PER_RUN,
            "main_design": "10 paired seeds; N=8,12,20; stable/recovery; six policies",
            "parameter_design": "10 seeds; eta in {0.03,0.07,0.11}; eps in {0.020,0.035,0.050}",
            "deadline_design": "10 seeds; deadlines 0.35,0.45,0.55 ms; Class-Top-K, RR, AdaSelect",
            "controlled_design": "10 paired seeds; 5 priority variants and 3 robustness scenarios x 4 policies",
            "rr_schedule": "independent shuffled worker order and pointer for each task class",
            "rand_obs": "uniform random selection from the same eligible candidate set as AdaSelect",
            "async_recovery": "independent start/end windows for each worker's specialist class",
            "rpc_delay": "0.5 ms sleep before each allocation RPC; calibration excluded",
            "task_digest": "same PBKDF2 task digest for every worker; capability and impairment costs use a separate dummy salt",
        })
        (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"WROTE {raw_path} rows={len(rows)} tasks={sum(int(r['tasks']) for r in rows)}", flush=True)
    finally:
        cluster.close()
        raw_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/mnt/data/AdaSelect_reproducible/results")
    parser.add_argument("--quick", action="store_true", help="run a small smoke-test matrix")
    parser.add_argument("--resume", action="store_true", help="resume from an existing CSV in --out")
    args = parser.parse_args()
    run_suite.resume = args.resume
    run_suite(Path(args.out), quick=args.quick)
