from karl_agent import KARL
import numpy as np
import time as time_module
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import random
import argparse
import sys

# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
args = vars(ap.parse_args())

g_type = "GMM"

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: safely print a section banner
# ──────────────────────────────────────────────────────────────────────────────
def banner(title):
    print("\n" + "=" * 65)
    print("  " + title)
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Parameter count
#  Prints a per-component breakdown of all trainable parameters in KARL.
#  This goes directly into the "Parameter Count" row of Table 2.
# ──────────────────────────────────────────────────────────────────────────────
def print_parameter_count(dqn):
    banner("SECTION 1 — Parameter Count Breakdown")

    net = dqn.karl_net

    def n(module):
        return sum(p.numel() for p in module.parameters())

    rows = [
        ("w_n2l  (input projection)",          net.w_n2l.numel()),
        ("KANneigh  — B-spline encoder",        n(net.p_node_conv_kan)),
        ("KANself   — B-spline encoder",        n(net.p_node_conv2_kan)),
        ("KANout    — B-spline encoder",        n(net.p_node_conv3_kan)),
        ("residual_mlp  (linear-ReLU branch)",  n(net.residual_mlp)),
        ("residual_alpha  (learnable scalar)",  net.residual_alpha.numel()),
        ("KANformer  (BitwiseMultipyLogis)",    n(net.layerNodeAttention_weight)),
        ("cross_product  (state-action proj.)", net.cross_product.numel()),
        ("w_layer1 + w_layer2  (softmax W)",    net.w_layer1.numel() + net.w_layer2.numel()),
        ("ChebyKAN_hidden  (decoder, d=4)",     n(net.kan_layer1) if net.kan_layer1 else 0),
        ("ChebyKAN_final   (decoder, d=4)",     n(net.kan_layer2)),
    ]

    total = sum(v for _, v in rows)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    col_w = 45
    print(f"\n  {'Component':<{col_w}} {'Params':>10}")
    print("  " + "-" * (col_w + 12))
    for label, count in rows:
        print(f"  {label:<{col_w}} {count:>10,}")
    print("  " + "-" * (col_w + 12))
    print(f"  {'TOTAL (all parameters)':<{col_w}} {total:>10,}")
    print(f"  {'TRAINABLE parameters':<{col_w}} {trainable:>10,}")
    print()

    return total, trainable


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Training time
#  Reads from MultiDismantler_torch's own timer (injected below via wrapper).
#  We time a single call to dqn.Fit() on the fly here.
#  This gives: milliseconds per gradient step at training scale (N=30-50).
# ──────────────────────────────────────────────────────────────────────────────
def measure_fit_time(dqn, n_steps=100):
    banner("SECTION 2 — Training Step Time  (Fit, N=30-50, batch=64)")

    # We need at least one graph in the replay buffer.
    # Gen a few graphs and play them to fill the buffer.
    print("  Filling replay buffer with synthetic graphs …")
    dqn.gen_new_graphs(30, 50)
    for _ in range(5):
        dqn.PlayGame(10, 1.0)   # eps=1 → pure random, fast

    # Warm-up: 10 steps not counted
    for _ in range(10):
        dqn.Fit()

    device = dqn.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_steps):
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            dqn.Fit()
            e.record()
            torch.cuda.synchronize(device)
            times.append(s.elapsed_time(e))       # milliseconds
        else:
            t0 = time_module.perf_counter()
            dqn.Fit()
            times.append((time_module.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    throughput = 1000.0 / mean_ms   # steps per second

    print(f"\n  Gradient steps timed:          {n_steps}")
    print(f"  Time per step (ms):            {mean_ms:.2f} ± {std_ms:.2f}")
    print(f"  Throughput (steps/sec):        {throughput:.1f}")
    print(f"  GPU:  {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print()

    return mean_ms, std_ms, throughput


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Inference time + peak GPU memory on real-world datasets
#  For each dataset we call EvaluateRealData exactly as before, but wrap it
#  with CUDA events (GPU) or perf_counter (CPU) and memory tracking.
#  This gives: per-step inference time and peak GPU memory for Table 2.
# ──────────────────────────────────────────────────────────────────────────────
def GetSolution(STEPRATIO, MODEL_FILE, save_dir, dqn):
    """
    Runs EvaluateRealData on all real-world benchmarks.
    Returns a list of result dicts, one per dataset.
    """
    data_test_path  = './data/real/'
    data_test_name  = [
        'CS-Aarhus_multiplex',
        'fao_trade_multiplex',
        'celegans_connectome_multiplex',
        'fb-tw',
        'homo_genetic_multiplex',
        'sacchpomb_genetic_multiplex',
        'Sanremo2016_final_multiplex',
    ]
    date_test_n     = [61, 214, 279, 1043, 18222, 4092, 56562]
    data_test_layer = [(1,2),(3,24),(2,3),(1,2),(1,2),(4,6),(1,2)]

    model_file = './models/{}'.format(MODEL_FILE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f'\n  Model : {model_file}')
    dqn.LoadModel(model_file)

    device = dqn.device
    all_results = []

    for j in range(len(data_test_name)):
        stepRatio  = STEPRATIO
        dname      = data_test_name[j]
        n_nodes    = date_test_n[j]
        data_test  = data_test_path + dname + '.edges'

        print(f'\n  ── Dataset: {dname}  (N={n_nodes}) ──')

        # ── Reset GPU memory counter before this dataset ──
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # ── Time the full dismantling run with CUDA events ──
        if device.type == 'cuda':
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end   = torch.cuda.Event(enable_timing=True)
            ev_start.record()
        else:
            _t0 = time_module.perf_counter()

        solution, solution_time_s, score = dqn.EvaluateRealData(
            model_file, data_test, save_dir, stepRatio, n_nodes,
            data_test_layer[j]
        )

        if device.type == 'cuda':
            ev_end.record()
            torch.cuda.synchronize(device)
            total_infer_ms   = ev_start.elapsed_time(ev_end)   # ms
            peak_mem_mb      = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            total_infer_ms   = (time_module.perf_counter() - _t0) * 1000
            peak_mem_mb      = float('nan')

        n_steps_taken        = max(len(solution), 1)
        per_step_ms          = total_infer_ms / n_steps_taken

        print(f'    AUDC              : {score:.6f}')
        print(f'    Nodes removed     : {n_steps_taken}')
        print(f'    Total infer time  : {total_infer_ms/1000:.2f} s  '
              f'({total_infer_ms:.1f} ms)')
        print(f'    Per-step infer    : {per_step_ms:.3f} ms/step')
        print(f'    Peak GPU memory   : {peak_mem_mb:.1f} MB')

        # ── Save per-dataset CSV (original behaviour) ──
        df = pd.DataFrame(
            np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))),
            index=['time', 'score'],
            columns=data_test_name
        )
        df.iloc[0, j] = solution_time_s
        df.iloc[1, j] = score
        save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)
        df.to_csv(
            save_dir_local + '/time&audc_%s.csv' % dname,
            encoding='utf-8', index=False
        )

        all_results.append({
            'dataset':          dname,
            'n_nodes':          n_nodes,
            'audc':             score,
            'n_steps':          n_steps_taken,
            'total_infer_ms':   total_infer_ms,
            'per_step_ms':      per_step_ms,
            'peak_mem_mb':      peak_mem_mb,
        })

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Summary table (paper-ready)
# ──────────────────────────────────────────────────────────────────────────────
def print_summary_table(total_params, trainable_params,
                        fit_mean_ms, fit_std_ms, throughput,
                        infer_results, device):
    banner("SECTION 4 — Paper-Ready Summary Table")

    gpu_str = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    print(f"\n  Hardware : {gpu_str}")
    print(f"  PyTorch  : {torch.__version__}\n")

    # ── Parameter table ──
    print(f"  ┌{'─'*55}┬{'─'*12}┐")
    print(f"  │ {'Metric':<53} │ {'Value':>10} │")
    print(f"  ├{'─'*55}┼{'─'*12}┤")
    print(f"  │ {'Total parameters':<53} │ {total_params:>10,} │")
    print(f"  │ {'Trainable parameters':<53} │ {trainable_params:>10,} │")
    print(f"  │ {'Training step time (ms) [N=30-50, batch=64]':<53} │ "
          f"{fit_mean_ms:>7.2f}±{fit_std_ms:<3.2f} │")
    print(f"  │ {'Training throughput (steps/sec)':<53} │ {throughput:>10.1f} │")
    print(f"  ├{'─'*55}┼{'─'*12}┤")

    for r in infer_results:
        lbl = f"Inference per-step ms  [{r['dataset']}  N={r['n_nodes']}]"
        print(f"  │ {lbl:<53} │ {r['per_step_ms']:>10.3f} │")

    print(f"  ├{'─'*55}┼{'─'*12}┤")

    for r in infer_results:
        lbl = f"Peak GPU mem MB  [{r['dataset']}  N={r['n_nodes']}]"
        val = f"{r['peak_mem_mb']:.1f}" if not np.isnan(r['peak_mem_mb']) else "  CPU"
        print(f"  │ {lbl:<53} │ {val:>10} │")

    print(f"  └{'─'*55}┴{'─'*12}┘")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Save everything to CSV
# ──────────────────────────────────────────────────────────────────────────────
def save_all_csv(save_dir, total_params, trainable_params,
                 fit_mean_ms, fit_std_ms, throughput,
                 infer_results):

    out_path = os.path.join(save_dir, 'computational_cost_summary.csv')

    rows = []

    # Parameter rows
    rows.append({'metric': 'total_parameters',
                 'dataset': 'ALL', 'n_nodes': '-',
                 'value': total_params, 'unit': 'count'})
    rows.append({'metric': 'trainable_parameters',
                 'dataset': 'ALL', 'n_nodes': '-',
                 'value': trainable_params, 'unit': 'count'})

    # Training rows
    rows.append({'metric': 'train_step_time_mean_ms',
                 'dataset': 'synthetic_GMM', 'n_nodes': '30-50',
                 'value': round(fit_mean_ms, 4), 'unit': 'ms'})
    rows.append({'metric': 'train_step_time_std_ms',
                 'dataset': 'synthetic_GMM', 'n_nodes': '30-50',
                 'value': round(fit_std_ms, 4), 'unit': 'ms'})
    rows.append({'metric': 'train_throughput_steps_per_sec',
                 'dataset': 'synthetic_GMM', 'n_nodes': '30-50',
                 'value': round(throughput, 2), 'unit': 'steps/sec'})

    # Inference + memory rows
    for r in infer_results:
        rows.append({'metric': 'inference_per_step_ms',
                     'dataset': r['dataset'], 'n_nodes': r['n_nodes'],
                     'value': round(r['per_step_ms'], 4), 'unit': 'ms/step'})
        rows.append({'metric': 'inference_total_ms',
                     'dataset': r['dataset'], 'n_nodes': r['n_nodes'],
                     'value': round(r['total_infer_ms'], 2), 'unit': 'ms'})
        rows.append({'metric': 'peak_gpu_memory_mb',
                     'dataset': r['dataset'], 'n_nodes': r['n_nodes'],
                     'value': round(r['peak_mem_mb'], 2), 'unit': 'MB'})
        rows.append({'metric': 'audc',
                     'dataset': r['dataset'], 'n_nodes': r['n_nodes'],
                     'value': round(r['audc'], 6), 'unit': 'dimensionless'})

    df = pd.DataFrame(rows, columns=['metric', 'dataset', 'n_nodes', 'value', 'unit'])
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"  ✓  Full summary saved to: {out_path}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    save_dir        = args['output']
    model_file_ckpt = 'g0-1_10w_TORCH-Model_{}_30_100/best_model.ckpt'.format(g_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ── Instantiate the agent ──
    dqn = KARL()

    # ── SECTION 1: parameter count ──
    total_params, trainable_params = print_parameter_count(dqn)

    # ── SECTION 2: training step time ──
    fit_mean_ms, fit_std_ms, throughput = measure_fit_time(dqn, n_steps=100)

    # ── SECTION 3: inference on real datasets ──
    banner("SECTION 3 — Inference Time & GPU Memory on Real-World Datasets")
    infer_results = GetSolution(
        STEPRATIO=0,
        MODEL_FILE=model_file_ckpt,
        save_dir=save_dir,
        dqn=dqn,
    )

    # ── SECTION 4: print the paper-ready table ──
    print_summary_table(
        total_params, trainable_params,
        fit_mean_ms, fit_std_ms, throughput,
        infer_results,
        dqn.device,
    )

    # ── SECTION 5: save CSV ──
    banner("SECTION 5 — Saving CSV")
    save_all_csv(
        save_dir,
        total_params, trainable_params,
        fit_mean_ms, fit_std_ms, throughput,
        infer_results,
    )


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    main()