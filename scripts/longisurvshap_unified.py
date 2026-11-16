"""Unified DynSurvX experiments for simulation, MSK, PBC and CoxSig/DDH/DySurv.

Place this script in the project's scripts/ directory and the JSON alongside it.
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ('scripts', 'notebooks') else SCRIPT_DIR
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / 'src')]

sns.set(style='whitegrid', font='STIXGeneral', context='talk', palette='colorblind')


# ============================================================
# Configuration and common data utilities
# ============================================================

def load_config(config_path, dataset, model, prediction_function):
    with open(config_path, encoding='utf-8') as f:
        raw = json.load(f)
    mode = {'survival': 'mean', 'med-survival': 'med', 'mean': 'mean', 'med': 'med'}[prediction_function]
    try:
        exp = raw['experiments'][dataset][model][mode]
    except KeyError as exc:
        raise ValueError(f'Unknown experiment: {dataset}/{model}/{mode}') from exc
    return raw['defaults'], raw['simulation'], exp, mode


def take_rows(a, idx):
    if a is None:
        return None
    if torch.is_tensor(a):
        idx = torch.as_tensor(idx, dtype=torch.long, device=a.device)
    return a[idx]


def to_numpy(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def load_dataset(dataset, sim_cfg, figure_dir):
    if dataset == 'MSK':
        from examples.data.MSK import load_MSK
        paths, labels, longi_names, static, static_names, info = load_MSK.load()
    elif dataset == 'PBC':
        from examples.data.PBC import load_PBC_Seq
        paths, labels, longi_names, static, static_names, info = load_PBC_Seq.load()
    else:
        from examples.data.longi_simulation import longi_sim
        s = sim_cfg
        paths, labels, static, _, info, _, _ = longi_sim.Simulation().load(
            n_samples=s['n_samples'], n_sampling_times=s['n_sampling_times'],
            end_time=s['end_time'], dim=s['n_longi_feats'] + 1,
            intercept=True, threshold=s['threshold'],
            coefs=torch.tensor(s['coefs'], dtype=torch.float32),
            coefs_mask=torch.tensor(s['coefs_mask'], dtype=torch.float32),
            important_feat_mask=torch.tensor(s['important_feat_mask'], dtype=torch.float32),
            theta=torch.tensor(s['theta'], dtype=torch.float32),
            hurst_list=s['hurst_list'], n_static_feats=s['n_static_feats'],
            save_path=str(figure_dir / 'simu_data.pdf'), show=False,
        )
        longi_names = [f'X{i+1}' for i in range(s['n_longi_feats'])]
        static_names = [f'W{i+1}' for i in range(s['n_static_feats'])]

    return dict(paths=paths, labels=np.asarray(labels), static=None if static is None else np.asarray(static),
                longi_names=list(longi_names), static_names=list(static_names), ddh_info=info)


def split_data(n, seed, train_share=.8, val_ratio=.1):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(train_share * n)
    train, test = idx[:n_train], idx[n_train:]
    n_fit = int((1 - val_ratio) * len(train))
    return dict(train=train, test=test, fit=train[:n_fit], val=train[n_fit:])


def prediction_grid(labels, exp):
    event_times = labels[labels[:, 1] == 1, 0]
    q = np.asarray(exp['quantile_pred_times'], dtype=float)
    pred_times = np.quantile(event_times, q)
    eval_times = np.asarray([
        np.max(np.quantile(event_times, np.minimum(q + (k + 1) * exp['eval_quantile_step'], 1.0)) - pred_times)
        for k in range(exp['n_eval_times'])
    ])
    return pred_times, eval_times


# ============================================================
# Model adapters: CoxSig
# ============================================================

def fit_coxsig(data, split, exp):
    from examples.learner.coxsig import CoxSignature
    model = CoxSignature(**exp['model_params'])
    model.train(take_rows(data['paths'], split['train']), data['labels'][split['train']],
                take_rows(data['static'], split['train']), plot_loss=False)
    return model, {}


def score_coxsig(model, state, data, split, pred_times, eval_times):
    return model.score(take_rows(data['paths'], split['test']), data['labels'][split['test']],
                       pred_times, eval_times, 'c_index',
                       static_feat=take_rows(data['static'], split['test'])), None


def plot_coxsig_importance(model, data, exp, outpath):
    order = exp['model_params']['sig_level']
    n_longi = to_numpy(data['paths']).shape[-1] - 1
    combs = [c for r in range(1, order + 1) for c in product(range(n_longi + 1), repeat=r)]
    coefs = np.asarray(model.model.coefs).reshape(-1)
    rows = []
    for i, name in enumerate(data['longi_names'], 1):
        pure = [combs.index((i,) * r) for r in range(1, order + 1)]
        cross = [j for j, c in enumerate(combs) if i in c and c not in [(i,) * r for r in range(1, order + 1)]]
        rows.extend([(name, 'Pure', np.linalg.norm(coefs[pure])),
                     (name, 'Cross', np.linalg.norm(coefs[cross]))])
    n_static = 0 if data['static'] is None else data['static'].shape[1]
    for j in range(n_static):
        rows.append((data['static_names'][j], 'Pure', coefs[-n_static + j] ** 2))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=pd.DataFrame(rows, columns=['Feat', 'Type', 'Val']), x='Feat', y='Val', hue='Type', ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Model adapters: Dynamic DeepHit
# ============================================================

def fit_ddh(data, split, exp, dataset, result_dir, force_retrain=False):
    from examples.learner.dynamic_deephit_ext import construct_df, Dynamic_DeepHit_ext
    cont, binary, scale, bin_df = data['ddh_info']
    paths_torch = torch.as_tensor(to_numpy(data['paths']), dtype=torch.float32).clone()
    labels = data['labels']
    df = construct_df(paths_torch, labels, cont, binary, scale, bin_df)
    model = Dynamic_DeepHit_ext()
    (x, times, y), (m1, m2, m3), x_mi = model.preprocess(df, cont, binary)
    model.sampling_times = to_numpy(data['paths'])[0, :, 0]
    model.ddh_info_sup = data['ddh_info']
    tr = split['train']
    train_data = (x[tr], x_mi[tr], y[tr], times[tr], m1[tr], m2[tr], m3[tr])
    ckpt = result_dir / 'dynamic_deephit_ckpt' / dataset
    ckpt.mkdir(parents=True, exist_ok=True)
    is_trained = (ckpt / 'model.weights.h5').exists() and not force_retrain and exp['model_params'].get('reuse_checkpoint', True)
    model.train(train_data, is_trained=is_trained, ckpt_dir=str(ckpt), plot_loss=False)
    return model, dict(x=x, x_mi=x_mi, scale=scale)


def score_ddh(model, state, data, split, pred_times, eval_times):
    from examples.learner.utils import score
    te = split['test']
    pred = model.predict(state['x'][te], state['x_mi'][te],
                         (pred_times * state['scale']).astype(int),
                         (eval_times * state['scale']).astype(int))
    labels = data['labels'][te]
    ci = np.full((len(pred_times), len(eval_times)), np.nan)
    bs = ci.copy()
    for j, t in enumerate(pred_times):
        keep = labels[:, 0] >= t
        if not keep.any():
            continue
        residual = np.column_stack([labels[keep, 0] - t, labels[keep, 1]])
        ci[j] = score('c_index', residual, residual, pred[:, j][keep], eval_times)
        bs[j] = score('bs', residual, residual, pred[:, j][keep], eval_times)
    return ci, bs


# ============================================================
# Model adapters: DySurv
# ============================================================

def fit_dysurv(data, split, exp, dataset, result_dir, force_retrain=False):
    import torchtuples as tt
    from pycox.models import LogisticHazard
    from examples.learner.DySurv.model_utils import DySurv, Loss
    from examples.learner.dysurv_ext import DySurvWrapper, make_dysurv_input, add_duration_channel

    paths = to_numpy(data['paths'])
    static = data['static']
    labels = data['labels']
    fit_idx, val_idx = split['fit'], split['val']
    x_fit = make_dysurv_input(paths[fit_idx], take_rows(static, fit_idx))
    x_val = make_dysurv_input(paths[val_idx], take_rows(static, val_idx))
    p = exp['model_params']
    labtrans = LogisticHazard.label_transform(p['num_durations'])
    y_fit = labtrans.fit_transform(labels[fit_idx, 0], labels[fit_idx, 1].astype(int))
    y_val = labtrans.transform(labels[val_idx, 0], labels[val_idx, 1].astype(int))
    train_input = add_duration_channel(x_fit, y_fit)
    val_input = add_duration_channel(x_val, y_val)
    train = tt.tuplefy(train_input, (y_fit, train_input))
    val = tt.tuplefy(val_input, (y_val, val_input))
    net = DySurv(in_features=x_fit.shape[2], encoded_features=p['latent_dim'],
                 out_features=labtrans.out_features, seq_len=x_fit.shape[1])
    model = LogisticHazard(net, tt.optim.Adam(p['learning_rate']),
                           duration_index=labtrans.cuts, loss=Loss([1.0, 0.1, 0.01]))
    ckpt_dir = result_dir / 'dysurv_ckpt' / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / 'model.pt'
    if ckpt.exists() and not force_retrain:
        saved = torch.load(ckpt, map_location='cpu', weights_only=False)
        if not np.allclose(np.asarray(saved['duration_index']), labtrans.cuts):
            raise ValueError('DySurv checkpoint duration cuts mismatch; use --force-retrain')
        model.net.load_state_dict(saved['model_state_dict'])
    else:
        model.fit(*train, batch_size=p['batch_size'], epochs=p['epochs'], val_data=val,
                  callbacks=[tt.cb.EarlyStopping(patience=p['patience'])], verbose=False)
        torch.save({'model_state_dict': model.net.state_dict(),
                    'duration_index': np.asarray(labtrans.cuts)}, ckpt)
    model.net.eval()
    return DySurvWrapper(model), {}


def score_dysurv(model, state, data, split, pred_times, eval_times):
    from lifelines import KaplanMeierFitter
    labels = data['labels']
    censor_km = KaplanMeierFitter().fit(labels[split['train'], 0],
                                       event_observed=1 - labels[split['train'], 1].astype(int))
    paths = take_rows(data['paths'], split['test'])
    static = take_rows(data['static'], split['test'])
    te_labels = labels[split['test']]
    ci = np.asarray([model.score(paths, te_labels, t, eval_times, metric='c_index', static_feat=static)
                     for t in pred_times])
    bs = np.asarray([model.score(paths, te_labels, t, eval_times, metric='bs',
                                 static_feat=static, censor_km=censor_km) for t in pred_times])
    return ci, bs


# ============================================================
# Common plotting, SHAP explanations, checkpointing
# ============================================================

def plot_performance(cindex, pred_times, eval_times, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cindex, annot=True, cmap='viridis', ax=ax)
    ax.set_xticklabels([f'{t:.2f}' for t in eval_times])
    ax.set_yticklabels([f'{t:.2f}' for t in pred_times])
    ax.set(xlabel=r'$\delta_t$', ylabel=r'$t_P$', title='C-index')
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def save_progress(outdir, results, success, failed, errors):
    np.save(outdir / 'shap.npy', np.asarray(results, dtype=object), allow_pickle=True)
    np.save(outdir / 'selected_indices.npy', np.asarray(success, dtype=object), allow_pickle=True)
    np.save(outdir / 'failed_indices.npy', np.asarray(failed, dtype=object), allow_pickle=True)
    pd.DataFrame(errors, columns=['pred_time_index', 'source_index', 'error_type', 'message']).to_csv(
        outdir / 'shap_errors.csv', index=False)


def explain_dataset(model, model_name, data, pred_times, exp, defaults, outdir):
    from survinsights.longi_explainer import explainer
    from survinsights.local_explaination._survlongishap import survshap_longi
    paths, labels, static = data['paths'], data['labels'], data['static']
    sample_times = to_numpy(paths)[0, :, 0]
    dt = float(sample_times[1] - sample_times[0])
    seed = defaults['seed']
    shap_cfg = exp['shap']
    shap_kwargs = dict(prediction_type=exp['prediction_type'], n_split=shap_cfg['n_split'],
                       intercept=shap_cfg['intercept'])
    for key in ('reg', 'delta_1', 'delta_2'):
        if shap_cfg.get(key) is not None:
            shap_kwargs[key] = shap_cfg[key]

    results, success, failed, errors = [], [], [], []
    for j, t in enumerate(pred_times):
        eligible = np.flatnonzero(labels[:, 0] > t + defaults['survival_margin_dt'] * dt)
        k = min(defaults['n_explain'], int(defaults['explain_fraction'] * len(eligible)))
        print(f'Prediction {j+1}/{len(pred_times)} t={t:.3f}: eligible={len(eligible)}, selected={k}')
        if not k:
            results.append([]); success.append(np.array([], dtype=int)); failed.append(np.array([], dtype=int))
            save_progress(outdir, results, success, failed, errors)
            continue
        local = np.random.default_rng(seed + j).choice(len(eligible), size=k, replace=False)
        sel_paths = take_rows(paths, eligible)
        sel_labels = labels[eligible]
        sel_static = take_rows(static, eligible)
        model_explainer = explainer(model, {'CoxSig':'coxsig','DDH':'DDH','DySurv':'dysurv'}[model_name],
                                    sel_paths, sel_labels, sel_static)
        shap_results, good, bad = [], [], []
        for count, i in enumerate(local, 1):
            i = int(i)
            source_idx = int(eligible[i])
            try:
                value = survshap_longi(model_explainer, i, take_rows(sel_paths, i),
                                       take_rows(sel_static, i), pred_time=np.array([t]),
                                       seed=seed + i, **shap_kwargs)
            except (IndexError, TypeError) as exc:
                bad.append(source_idx)
                errors.append((j, source_idx, type(exc).__name__, str(exc)))
                print(f'  Skipped {source_idx}: {type(exc).__name__}: {exc}')
                continue
            shap_results.append(value)
            good.append(source_idx)
            if count % defaults['progress_every'] == 0 or count == k:
                print(f'  {count}/{k}, success={len(good)}, failed={len(bad)}')
        results.append(shap_results)
        success.append(np.asarray(good, dtype=int))
        failed.append(np.asarray(bad, dtype=int))
        save_progress(outdir, results, success, failed, errors)
    return results, success, failed


# ============================================================
# Main experiment and CLI
# ============================================================

def main(model='CoxSig', dataset='simu', prediction_function='survival',
         config_path=None, force_retrain=False, run_explanations=True):
    config_path = Path(config_path) if config_path else SCRIPT_DIR / 'longisurvshap_experiments.json'
    defaults, sim_cfg, exp, mode = load_config(config_path, dataset, model, prediction_function)
    np.random.seed(defaults['seed'])
    torch.manual_seed(defaults['seed'])
    outdir = PROJECT_ROOT / 'results' / 'longisurvshap' / dataset / model / mode
    figdir = PROJECT_ROOT / 'figures' / 'longisurvshap' / dataset / model / mode
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(dict(dataset=dataset, model=model, mode=mode, defaults=defaults,
                       simulation=sim_cfg if dataset == 'simu' else None, experiment=exp), f, indent=2)

    data = load_dataset(dataset, sim_cfg, figdir)
    split = split_data(len(data['labels']), defaults['seed'], defaults['train_share'])
    pred_times, eval_times = prediction_grid(data['labels'], exp)
    print('Prediction times:', pred_times, 'Evaluation horizons:', eval_times)
    np.savez(outdir / 'experiment_grid.npz', pred_times=pred_times, eval_times=eval_times,
             train_index=split['train'], test_index=split['test'], fit_index=split['fit'], val_index=split['val'])

    fitters = {'CoxSig': fit_coxsig, 'DDH': fit_ddh, 'DySurv': fit_dysurv}
    scorers = {'CoxSig': score_coxsig, 'DDH': score_ddh, 'DySurv': score_dysurv}
    start = time.time()
    if model == 'CoxSig':
        trained, state = fitters[model](data, split, exp)
        plot_coxsig_importance(trained, data, exp, figdir / 'signature_importance.pdf')
    else:
        trained, state = fitters[model](data, split, exp, dataset, PROJECT_ROOT / 'results', force_retrain)
    print(f'Model training/loading: {time.time() - start:.1f}s')
    ci, bs = scorers[model](trained, state, data, split, pred_times, eval_times)
    np.save(outdir / 'c_index.npy', ci)
    if bs is not None:
        np.save(outdir / 'brier_score.npy', bs)
    plot_performance(ci, pred_times, eval_times, figdir / 'c_index.pdf')
    if run_explanations:
        explain_dataset(trained, model, data, pred_times, exp, defaults, outdir)
    print(f'Finished {dataset}/{model}/{mode}. Results: {outdir}')
    return outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['CoxSig', 'DDH', 'DySurv'], default='CoxSig')
    parser.add_argument('--dataset', choices=['simu', 'MSK', 'PBC'], default='simu')
    parser.add_argument('--prediction-function', choices=['survival', 'med-survival'], default='survival')
    parser.add_argument('--config', default=str(SCRIPT_DIR / 'longisurvshap_experiments.json'))
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--no-explanations', action='store_true')
    args = parser.parse_args()
    main(args.model, args.dataset, args.prediction_function, args.config,
         args.force_retrain, not args.no_explanations)
