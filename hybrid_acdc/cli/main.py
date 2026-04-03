from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hybrid_acdc.utils.io import load_yaml, save_json, load_json
from hybrid_acdc.plant.avg_model import simulate_averaged
from hybrid_acdc.cost.performance_index import compute_cost, CostWeights
from hybrid_acdc.controllers.ofo import OFOConfig, make_delta, spsa_update
from hybrid_acdc.controllers.control_laws import OuterLoopConfig, PIDOuterController, OFONNOuterController
from hybrid_acdc.controllers.nn_disturbance import DisturbanceNNConfig
from hybrid_acdc.plant.vsc_switching import simulate_switching_window, compute_thd


def _make_eval_cfg(cfg: dict) -> dict:
    """Return a lightweight config for tuning/optimization.

    We intentionally evaluate the cost with a coarser dt and shorter horizon
    (still representing a compressed day) to keep autotune/OFO practical.
    """
    c = dict(cfg)
    sim = dict(cfg.get('sim', {}))
    sim['dt'] = float(sim.get('eval_dt', sim.get('dt', 0.002)))
    sim['t_end'] = float(sim.get('eval_t_end', sim.get('t_end', 240.0)))
    sim['compress_24h_to_seconds'] = float(sim['t_end'])
    c['sim'] = sim
    return c


def _extract_outer(cfg: dict) -> OuterLoopConfig:
    dc = cfg['dc_bus']
    sec = cfg['secondary']
    vdc_nom = float(dc['vdc_nom'])
    P_import_max = float(sec.get('P_import_max_kw', 500.0)) * 1e3
    P_export_max = float(sec.get('P_export_max_kw', 500.0)) * 1e3
    return OuterLoopConfig(vdc_nom=vdc_nom, P_import_max=P_import_max, P_export_max=P_export_max)


def cmd_autotune(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    cfg_eval = _make_eval_cfg(cfg)
    outer = _extract_outer(cfg)

    grid = cfg.get('grid', {})
    vac_bounds = (
        float(grid.get('v_phase_rms_min', 220.0)),
        float(grid.get('v_phase_rms_max', 240.0)),
    )

    rng = np.random.default_rng(int(cfg['sim'].get('seed', 7)))
    weights = CostWeights()

    # search space (wide enough for stiff DC-bus dynamics when Cdc is small)
    # With Cdc=0.02 F and V~850 V, the plant gain is ~1/(C*V)=0.0588 V/(W*s),
    # so practical PI gains can easily exceed 25.
    kp_lo, kp_hi = (1.0, 2000.0)
    ki_lo, ki_hi = (10.0, 200000.0)
    alpha = float(cfg.get('secondary', {}).get('P_share_alpha', 0.5))

    best = {'cost': float('inf'), 'kp': None, 'ki': None, 'alpha': alpha}

    # Deterministic warm-starts (PI tuning for an integrator plant):
    # Vdot = (1/(Cdc*Vdc)) * p_net  => gain k = 1/(Cdc*V)
    k_dc = 1.0 / (float(cfg_eval['dc_bus']['Cdc']) * float(cfg_eval['dc_bus']['vdc_nom']))
    w_candidates = [0.5, 1.0, 2.0, 4.0, 6.0]
    candidates = [(w / k_dc, (w ** 2) / k_dc) for w in w_candidates]

    # Add a few log-uniform random candidates for broader exploration
    for _ in range(max(0, int(args.max_evals) - len(candidates))):
        kp = float(10 ** rng.uniform(np.log10(kp_lo), np.log10(kp_hi)))
        ki = float(10 ** rng.uniform(np.log10(ki_lo), np.log10(ki_hi)))
        candidates.append((kp, ki))

    for i, (kp, ki) in enumerate(candidates, start=1):
        ctrl = PIDOuterController(kp=kp, ki=ki, alpha=alpha, outer=outer)
        sig = simulate_averaged(cfg_eval, controller=ctrl, seed=int(cfg['sim'].get('seed', 7)))
        J = compute_cost(
            sig,
            vdc_bounds=(cfg['dc_bus']['vdc_min'], cfg['dc_bus']['vdc_max']),
            vac_bounds=vac_bounds,
            weights=weights,
        )

        if J < best['cost']:
            best = {'cost': float(J), 'kp': kp, 'ki': ki, 'alpha': alpha}

        if args.verbose and (i == 1 or i % 5 == 0 or i == len(candidates)):
            print(f"[PID autotune] eval {i:03d}/{len(candidates)}: cost={J:.4f}  (best={best['cost']:.4f})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = {'vdc_kp': best['kp'], 'vdc_ki': best['ki'], 'alpha': best['alpha'], 'cost': best['cost']}
    save_json(ref, out / 'pid_ref.json')
    print(f"Saved PID reference gains to: {out / 'pid_ref.json'}")
    return 0


def _run_with_theta(cfg: dict, theta: np.ndarray, controller_kind: str) -> tuple[float, dict]:
    outer = _extract_outer(cfg)
    kp, ki, alpha = map(float, theta)

    if controller_kind == 'ofo_nn':
        ctrl = OFONNOuterController(kp=kp, ki=ki, alpha=alpha, outer=outer, nn_cfg=DisturbanceNNConfig())
    else:
        ctrl = PIDOuterController(kp=kp, ki=ki, alpha=alpha, outer=outer)

    cfg_eval = _make_eval_cfg(cfg)
    sig = simulate_averaged(cfg_eval, controller=ctrl, seed=int(cfg['sim'].get('seed', 7)))
    grid = cfg.get('grid', {})
    vac_bounds = (
        float(grid.get('v_phase_rms_min', 220.0)),
        float(grid.get('v_phase_rms_max', 240.0)),
    )
    J = compute_cost(
        sig,
        vdc_bounds=(cfg['dc_bus']['vdc_min'], cfg['dc_bus']['vdc_max']),
        vac_bounds=vac_bounds,
        weights=CostWeights(),
    )

    summary = {
        'cost': float(J),
        'theta': {'vdc_kp': kp, 'vdc_ki': ki, 'alpha': alpha},
    }
    return float(J), summary


def cmd_iterate(args: argparse.Namespace) -> int:
    cfg = load_yaml(args.config)
    pid_ref = load_json(args.pid_ref)
    theta_ref = np.array([pid_ref['vdc_kp'], pid_ref['vdc_ki'], pid_ref.get('alpha', 0.5)], dtype=float)

    # initial theta (start away from reference to make convergence visible)
    theta = theta_ref.copy()
    theta[0] *= 0.35
    theta[1] *= 0.35
    theta[2] = float(np.clip(theta_ref[2] + 0.05, 0.05, 0.95))

    ofo_cfg = OFOConfig(n_iter=args.iters, step_size=args.step_size, perturb=args.perturb, seed=int(cfg['sim'].get('seed', 7)))
    rng = np.random.default_rng(ofo_cfg.seed)

    history = []

    for k in range(1, ofo_cfg.n_iter + 1):
        delta = make_delta(rng)
        theta_plus = theta + ofo_cfg.perturb * delta
        theta_minus = theta - ofo_cfg.perturb * delta

        J_plus, _ = _run_with_theta(cfg, theta_plus, controller_kind=args.controller)
        J_minus, _ = _run_with_theta(cfg, theta_minus, controller_kind=args.controller)

        theta = spsa_update(theta, J_plus, J_minus, delta, cfg=ofo_cfg)

        # evaluate current
        J, summary = _run_with_theta(cfg, theta, controller_kind=args.controller)

        err_to_ref = float(np.linalg.norm(theta - theta_ref) / (np.linalg.norm(theta_ref) + 1e-9))
        row = {
            'iter': k,
            'mean_cost': J,
            'theta': summary['theta'],
            'err_to_pid_ref': err_to_ref,
        }
        history.append(row)

        if args.verbose:
            print(
                f"[{args.controller}] iter {k:03d}/{ofo_cfg.n_iter}: mean_cost={J:.4f}  "
                f"theta(vdc_kp)={theta[0]:.3f}  theta(vdc_ki)={theta[1]:.1f}  alpha={theta[2]:.3f}  "
                f"err_to_ref={err_to_ref:.4f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json({'pid_ref': pid_ref, 'history': history}, out / 'iterate_history.json')
    print(f"Saved {args.controller} iteration history to: {out / 'iterate_history.json'}")
    return 0


def cmd_thd(args: argparse.Namespace) -> int:
    """Compute current THD in a short switching window at a representative operating point.

    For the paper, this is used for the robustness test (e.g., Lf = 30 mH vs 15 mH).
    """
    cfg = load_yaml(args.config)
    outer = _extract_outer(cfg)

    # Load PID ref (used when controller=pid or as baseline starting point)
    pid_ref = load_json(args.pid_ref)
    theta = np.array([pid_ref['vdc_kp'], pid_ref['vdc_ki'], pid_ref.get('alpha', 0.5)], dtype=float)
    kp, ki, alpha = map(float, theta)

    if args.controller == 'ofo_nn':
        ctrl = OFONNOuterController(kp=kp, ki=ki, alpha=alpha, outer=outer, nn_cfg=DisturbanceNNConfig())
    else:
        ctrl = PIDOuterController(kp=kp, ki=ki, alpha=alpha, outer=outer)

    # Full (non-eval) averaged sim to get an operating point
    sig = simulate_averaged(cfg, controller=ctrl, seed=int(cfg['sim'].get('seed', 7)))

    # Choose a representative time index (midday ~ hour 12)
    t_end = float(cfg['sim']['t_end'])
    t_target = 0.5 * t_end
    k = int(np.argmin(np.abs(sig.t - t_target)))

    grid = cfg.get('grid', {})
    v_phase_rms = float(grid.get('v_phase_rms_nom', 230.0))
    f_grid = float(cfg['sim'].get('f_grid', 50.0))

    def run_one(L: float) -> float:
        """Run a short switching-window sim and compute THD.

        This is a simplified validation step: we build a sinusoidal current reference
        whose RMS magnitude matches the averaged-model operating point.
        """
        vdc = float(sig.vdc[k])

        # Build a 3-phase sinusoidal current reference
        t_emt = np.arange(0.0, float(args.t_emt), float(args.dt_emt))
        w = 2.0 * np.pi * float(f_grid)
        # Use VSC-1 apparent current magnitude as representative
        i_mag = float(np.sqrt(sig.id1[k] ** 2 + sig.iq1[k] ** 2))
        ia = i_mag * np.sin(w * t_emt)
        ib = i_mag * np.sin(w * t_emt - 2.0 * np.pi / 3.0)
        ic = i_mag * np.sin(w * t_emt + 2.0 * np.pi / 3.0)
        i_ref_abc = np.vstack([ia, ib, ic]).T

        out = simulate_switching_window(
            vdc=vdc,
            v_phase_rms=float(v_phase_rms),
            f_grid=float(f_grid),
            f_sw=float(args.f_sw),
            L=float(L),
            R=float(args.Rf),
            i_ref_abc=i_ref_abc,
            dt=float(args.dt_emt),
            t_end=float(args.t_emt),
        )
        thd = compute_thd(
            out['i_abc'],
            f_grid=float(f_grid),
            dt=float(args.dt_emt),
            n_harmonics=int(args.n_harm),
        )
        return float(thd)

    thd_30 = run_one(float(args.L1))
    thd_15 = run_one(float(args.L2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            'operating_point': {
                't_s': float(sig.t[k]),
                'vdc': float(sig.vdc[k]),
                'id1': float(sig.id1[k]),
                'iq1': float(sig.iq1[k]),
            },
            'switching': {
                'f_sw': float(args.f_sw),
                'dt_emt': float(args.dt_emt),
                't_emt': float(args.t_emt),
                'Rf': float(args.Rf),
                'L1': float(args.L1),
                'L2': float(args.L2),
                'n_harm': int(args.n_harm),
            },
            'thd': {
                'L1': thd_30,
                'L2': thd_15,
            },
        },
        out / 'thd_summary.json',
    )
    print(f"Saved THD summary to: {out / 'thd_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='hybrid_acdc', description='Hybrid AC/DC OFO / OFO-NN scaffold')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_auto = sub.add_parser('autotune', help='Auto-tune PID reference gains')
    p_auto.add_argument('--config', required=True)
    p_auto.add_argument('--out', required=True)
    p_auto.add_argument('--max_evals', type=int, default=48)
    p_auto.add_argument('--verbose', action='store_true')
    p_auto.set_defaults(func=cmd_autotune)

    p_it = sub.add_parser('iterate', help='Run OFO or OFO-NN iterations')
    p_it.add_argument('--config', required=True)
    p_it.add_argument('--pid_ref', required=True)
    p_it.add_argument('--controller', choices=['ofo', 'ofo_nn'], default='ofo')
    p_it.add_argument('--out', required=True)
    p_it.add_argument('--iters', type=int, default=60)
    p_it.add_argument('--step_size', type=float, default=0.08)
    p_it.add_argument('--perturb', type=float, default=0.2)
    p_it.add_argument('--verbose', action='store_true')
    p_it.set_defaults(func=cmd_iterate)

    p_thd = sub.add_parser('thd', help='Compute switching-current THD at a representative operating point')
    p_thd.add_argument('--config', required=True)
    p_thd.add_argument('--pid_ref', required=True)
    p_thd.add_argument('--controller', choices=['pid', 'ofo_nn'], default='ofo_nn')
    p_thd.add_argument('--out', required=True)
    p_thd.add_argument('--f_sw', type=float, default=10000.0)
    p_thd.add_argument('--dt_emt', type=float, default=2e-6)
    p_thd.add_argument('--t_emt', type=float, default=0.25)
    p_thd.add_argument('--Rf', type=float, default=0.06)
    p_thd.add_argument('--L1', type=float, default=30e-3)
    p_thd.add_argument('--L2', type=float, default=15e-3)
    p_thd.add_argument('--n_harm', type=int, default=40)
    p_thd.set_defaults(func=cmd_thd)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
