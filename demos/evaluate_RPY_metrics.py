# -*- coding: utf-8 -*-
#
# Compare recognized RPY metrics from DECA results against ground truth stored in
# step1.frame_data.json files next to source images.

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
FRAME_DATA_FILENAME = 'step1.frame_data.json'
EVAL_MARKER = '# --- RPY evaluation ---'
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'rpy_eval.config.json')
CONFIG_KEYS = {
    'inputpath', 'log_filename', 'frame_data_filename',
    'html_report_filename', 'visualize',
}


@dataclass
class RPYAngles:
    roll: float
    pitch: float
    yaw: float


@dataclass
class RPYFlags:
    roll: bool
    pitch: bool
    yaw: bool


@dataclass
class EvaluationResult:
    rel_path: str
    reference: RPYAngles
    recognized: RPYAngles
    abs_diff: RPYAngles
    same_axis_sign: RPYFlags


def normalize_input_path(path):
    return os.path.normpath(path.replace('\\', os.sep))


def collect_images_recursive(root):
    imagepaths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('result_')]
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                imagepaths.append(os.path.join(dirpath, filename))
    return sorted(imagepaths)


def parse_locale_float(value):
    return float(str(value).strip().replace(',', '.'))


def parse_rpy_string(rpy_string):
    parts = [part.strip() for part in str(rpy_string).split(';')]
    if len(parts) != 3:
        raise ValueError(f'expected 3 RPY values, got {len(parts)}: {rpy_string!r}')

    return RPYAngles(
        roll=parse_locale_float(parts[0]),
        pitch=parse_locale_float(parts[1]),
        yaw=parse_locale_float(parts[2]),
    )


def find_rpy_string_in_frame_data(data):
    rpy_metrics = data.get('RPYMetrics')
    if isinstance(rpy_metrics, dict):
        rpy_string = rpy_metrics.get('RPY')
        if rpy_string:
            return str(rpy_string)

    for metric in data.get('metrics', []):
        if not isinstance(metric, dict):
            continue
        for value in metric.get('values', []):
            if not isinstance(value, dict):
                continue
            rpy_metrics = value.get('RPYMetrics')
            if not isinstance(rpy_metrics, dict):
                continue
            rpy_string = rpy_metrics.get('RPY')
            if rpy_string:
                return str(rpy_string)

    raise KeyError('RPYMetrics.RPY not found in frame data')


def load_reference_rpy(frame_data_path):
    with open(frame_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rpy_string = find_rpy_string_in_frame_data(data)
    return parse_rpy_string(rpy_string)


RPY_LINE_PATTERNS = {
    'pitch': re.compile(r'^Pitch\s*\(X-axis\)\s*:\s*([-+0-9.,]+)\s*$', re.IGNORECASE),
    'yaw': re.compile(r'^Yaw\s*\(Y-axis\)\s*:\s*([-+0-9.,]+)\s*$', re.IGNORECASE),
    'roll': re.compile(r'^Roll\s*\(Z-axis\)\s*:\s*([-+0-9.,]+)\s*$', re.IGNORECASE),
}


def load_recognized_rpy(rpy_file_path):
    values = {}
    with open(rpy_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped == EVAL_MARKER:
                break
            for axis, pattern in RPY_LINE_PATTERNS.items():
                match = pattern.match(stripped)
                if match:
                    values[axis] = parse_locale_float(match.group(1))
                    break

    missing = [axis for axis in ('roll', 'pitch', 'yaw') if axis not in values]
    if missing:
        raise ValueError(f'missing recognized angles: {", ".join(missing)}')

    return RPYAngles(
        roll=values['roll'],
        pitch=values['pitch'],
        yaw=values['yaw'],
    )


def compare_axis_signs(reference, recognized):
    def signs_match(a, b):
        if a == 0 and b == 0:
            return True
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    return RPYFlags(
        roll=signs_match(reference.roll, recognized.roll),
        pitch=signs_match(reference.pitch, recognized.pitch),
        yaw=signs_match(reference.yaw, recognized.yaw),
    )


def abs_difference(reference, recognized):
    return RPYAngles(
        roll=abs(abs(reference.roll) - abs(recognized.roll)),
        pitch=abs(abs(reference.pitch) - abs(recognized.pitch)),
        yaw=abs(abs(reference.yaw) - abs(recognized.yaw)),
    )


def format_bool(value):
    return 'True' if value else 'False'


def append_evaluation_to_rpy_file(rpy_file_path, reference, recognized, abs_diff, signs):
    with open(rpy_file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    preserved_lines = []
    for line in original_lines:
        if line.strip() == EVAL_MARKER:
            break
        preserved_lines.append(line)

    while preserved_lines and not preserved_lines[-1].endswith('\n'):
        preserved_lines[-1] += '\n'

    evaluation_lines = [
        f'{EVAL_MARKER}\n',
        f'Reference Roll:  {reference.roll:.4f}\n',
        f'Reference Pitch: {reference.pitch:.4f}\n',
        f'Reference Yaw:   {reference.yaw:.4f}\n',
        f'|Roll| difference:  {abs_diff.roll:.4f}\n',
        f'|Pitch| difference: {abs_diff.pitch:.4f}\n',
        f'|Yaw| difference:   {abs_diff.yaw:.4f}\n',
        f'Roll SameAxisSign:  {format_bool(signs.roll)}\n',
        f'Pitch SameAxisSign: {format_bool(signs.pitch)}\n',
        f'Yaw SameAxisSign:   {format_bool(signs.yaw)}\n',
    ]

    with open(rpy_file_path, 'w', encoding='utf-8') as f:
        f.writelines(preserved_lines)
        f.writelines(evaluation_lines)


def evaluate_image(imagepath, dataset_root, frame_data_filename):
    image_dir = os.path.dirname(imagepath)
    name = os.path.splitext(os.path.basename(imagepath))[0]
    rel_path = os.path.relpath(imagepath, dataset_root)

    frame_data_path = os.path.join(image_dir, frame_data_filename)
    if not os.path.isfile(frame_data_path):
        raise FileNotFoundError(f'missing frame data: {frame_data_path}')

    rpy_file_path = os.path.join(image_dir, f'result_{name}', f'{name}_rpy.txt')
    if not os.path.isfile(rpy_file_path):
        raise FileNotFoundError(f'missing recognized metrics: {rpy_file_path}')

    reference = load_reference_rpy(frame_data_path)
    recognized = load_recognized_rpy(rpy_file_path)
    abs_diff = abs_difference(reference, recognized)
    signs = compare_axis_signs(reference, recognized)

    append_evaluation_to_rpy_file(rpy_file_path, reference, recognized, abs_diff, signs)

    return EvaluationResult(
        rel_path=rel_path,
        reference=reference,
        recognized=recognized,
        abs_diff=abs_diff,
        same_axis_sign=signs,
    )


def max_deviation(result):
    return max(result.abs_diff.roll, result.abs_diff.pitch, result.abs_diff.yaw)


def deviation_color(ratio):
    ratio = max(0.0, min(1.0, ratio))
    red = int(22 + (220 - 22) * ratio)
    green = int(163 + (45 - 163) * ratio)
    blue = int(74 + (45 - 74) * ratio)
    return f'rgb({red}, {green}, {blue})'


def html_escape(text):
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def write_html_report(report_path, dataset_root, results, skipped):
    if not results:
        html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>RPY deviation report</title>
</head>
<body>
  <h1>Визуализация отклонений RPY</h1>
  <p>Нет данных для визуализации.</p>
  <p>Пропущено изображений: {len(skipped)}</p>
</body>
</html>'''
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return

    sorted_results = sorted(results, key=max_deviation, reverse=True)
    max_value = max_deviation(sorted_results[0])
    min_value = max_deviation(sorted_results[-1])

    rows = []
    for rank, item in enumerate(sorted_results, start=1):
        score = max_deviation(item)
        if max_value > min_value:
            ratio = (score - min_value) / (max_value - min_value)
        else:
            ratio = 0.0
        color = deviation_color(ratio)
        image_href = html_escape(item.rel_path.replace(os.sep, '/'))
        rows.append(f'''
        <article class="card" style="border-color: {color}; box-shadow: 0 0 0 1px {color};">
          <div class="rank" style="background: {color};">{rank}</div>
          <img src="{image_href}" alt="{html_escape(item.rel_path)}" loading="lazy">
          <div class="content">
            <h2>{html_escape(item.rel_path)}</h2>
            <p class="score">Макс. отклонение: <strong>{score:.4f}°</strong></p>
            <div class="bars">
              <div class="bar-row">
                <span>Roll</span>
                <div class="bar"><span style="width:{(item.abs_diff.roll / max_value * 100) if max_value else 0:.2f}%; background:{deviation_color((item.abs_diff.roll - min_value) / (max_value - min_value) if max_value > min_value else 0)};"></span></div>
                <span>{item.abs_diff.roll:.4f}°</span>
              </div>
              <div class="bar-row">
                <span>Pitch</span>
                <div class="bar"><span style="width:{(item.abs_diff.pitch / max_value * 100) if max_value else 0:.2f}%; background:{deviation_color((item.abs_diff.pitch - min_value) / (max_value - min_value) if max_value > min_value else 0)};"></span></div>
                <span>{item.abs_diff.pitch:.4f}°</span>
              </div>
              <div class="bar-row">
                <span>Yaw</span>
                <div class="bar"><span style="width:{(item.abs_diff.yaw / max_value * 100) if max_value else 0:.2f}%; background:{deviation_color((item.abs_diff.yaw - min_value) / (max_value - min_value) if max_value > min_value else 0)};"></span></div>
                <span>{item.abs_diff.yaw:.4f}°</span>
              </div>
            </div>
            <details>
              <summary>Подробнее</summary>
              <p>Эталон: Roll={item.reference.roll:.4f}, Pitch={item.reference.pitch:.4f}, Yaw={item.reference.yaw:.4f}</p>
              <p>Распознано: Roll={item.recognized.roll:.4f}, Pitch={item.recognized.pitch:.4f}, Yaw={item.recognized.yaw:.4f}</p>
              <p>SameAxisSign: Roll={format_bool(item.same_axis_sign.roll)}, Pitch={format_bool(item.same_axis_sign.pitch)}, Yaw={format_bool(item.same_axis_sign.yaw)}</p>
            </details>
          </div>
        </article>''')

    skipped_html = ''
    if skipped:
        skipped_items = ''.join(
            f'<li>{html_escape(path)}: {html_escape(reason)}</li>'
            for path, reason in skipped
        )
        skipped_html = f'''
        <section class="skipped">
          <h2>Пропущено ({len(skipped)})</h2>
          <ul>{skipped_items}</ul>
        </section>'''

    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>RPY deviation report</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Segoe UI, sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .legend, .summary, .skipped {{
      background: #fff;
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }}
    .legend-bar {{
      height: 18px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgb(22,163,74), rgb(234,179,8), rgb(220,45,45));
      margin: 12px 0;
    }}
    .legend-labels {{
      display: flex;
      justify-content: space-between;
      font-size: 14px;
      color: #52606d;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fff;
      border-radius: 12px;
      overflow: hidden;
      border: 3px solid transparent;
      box-shadow: 0 1px 3px rgba(0,0,0,.08);
      display: flex;
      flex-direction: column;
    }}
    .rank {{
      color: #fff;
      font-weight: 700;
      padding: 8px 12px;
      font-size: 14px;
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #d9e2ec;
    }}
    .content {{
      padding: 14px 16px 18px;
    }}
    .score {{
      margin: 0 0 12px;
      font-size: 15px;
    }}
    .bars {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 42px 1fr 64px;
      gap: 8px;
      align-items: center;
      font-size: 13px;
    }}
    .bar {{
      height: 10px;
      background: #e4e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
    }}
    details {{
      font-size: 13px;
      color: #52606d;
    }}
    details p {{
      margin: 6px 0;
    }}
  </style>
</head>
<body>
  <h1>Визуализация отклонений RPY</h1>
  <section class="summary">
    <p>Датасет: <strong>{html_escape(dataset_root)}</strong></p>
    <p>Изображений в отчёте: <strong>{len(results)}</strong></p>
    <p>Сортировка: от наибольшего отклонения к наименьшему.</p>
  </section>
  <section class="legend">
    <h2>Градиент отклонения</h2>
    <div class="legend-bar"></div>
    <div class="legend-labels">
      <span>Наименьшее ({min_value:.4f}°)</span>
      <span>Наибольшее ({max_value:.4f}°)</span>
    </div>
  </section>
  <section class="grid">
    {''.join(rows)}
  </section>
  {skipped_html}
</body>
</html>'''

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ['true', '1']


def summarize_axis(values):
    if not values:
        return None
    return {
        'mean': statistics.mean(values),
        'max': max(values),
    }


def write_dataset_log(log_path, dataset_root, results, skipped):
    roll_diffs = [item.abs_diff.roll for item in results]
    pitch_diffs = [item.abs_diff.pitch for item in results]
    yaw_diffs = [item.abs_diff.yaw for item in results]

    roll_stats = summarize_axis(roll_diffs)
    pitch_stats = summarize_axis(pitch_diffs)
    yaw_stats = summarize_axis(yaw_diffs)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('RPY evaluation log\n')
        f.write(f'Dataset root: {dataset_root}\n')
        f.write(f'Processed images: {len(results)}\n')
        f.write(f'Skipped images: {len(skipped)}\n\n')

        if results:
            f.write('Per-image results:\n')
            for item in results:
                f.write(f'{item.rel_path}\n')
                f.write(
                    '  reference: '
                    f'Roll={item.reference.roll:.4f}, '
                    f'Pitch={item.reference.pitch:.4f}, '
                    f'Yaw={item.reference.yaw:.4f}\n'
                )
                f.write(
                    '  recognized: '
                    f'Roll={item.recognized.roll:.4f}, '
                    f'Pitch={item.recognized.pitch:.4f}, '
                    f'Yaw={item.recognized.yaw:.4f}\n'
                )
                f.write(
                    '  |difference|: '
                    f'Roll={item.abs_diff.roll:.4f}, '
                    f'Pitch={item.abs_diff.pitch:.4f}, '
                    f'Yaw={item.abs_diff.yaw:.4f}\n'
                )
                f.write(
                    '  SameAxisSign: '
                    f'Roll={format_bool(item.same_axis_sign.roll)}, '
                    f'Pitch={format_bool(item.same_axis_sign.pitch)}, '
                    f'Yaw={format_bool(item.same_axis_sign.yaw)}\n\n'
                )

            f.write('Summary statistics:\n')
            f.write(
                f'Roll  mean |difference|: {roll_stats["mean"]:.4f}, '
                f'max |difference|: {roll_stats["max"]:.4f}\n'
            )
            f.write(
                f'Pitch mean |difference|: {pitch_stats["mean"]:.4f}, '
                f'max |difference|: {pitch_stats["max"]:.4f}\n'
            )
            f.write(
                f'Yaw   mean |difference|: {yaw_stats["mean"]:.4f}, '
                f'max |difference|: {yaw_stats["max"]:.4f}\n'
            )
        else:
            f.write('No images were evaluated.\n')

        if skipped:
            f.write('\nSkipped:\n')
            for rel_path, reason in skipped:
                f.write(f'{rel_path}: {reason}\n')


def load_run_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f'config file must contain a JSON object: {config_path}')

    unknown_keys = set(config) - CONFIG_KEYS
    if unknown_keys:
        print(f'warning: ignoring unknown config keys: {", ".join(sorted(unknown_keys))}')

    return {key: config[key] for key in CONFIG_KEYS if key in config}


def resolve_config_path(config_path):
    if config_path:
        return os.path.abspath(config_path)
    if os.path.isfile(DEFAULT_CONFIG_PATH):
        return DEFAULT_CONFIG_PATH
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate recognized RPY metrics against step1.frame_data.json ground truth')

    parser.add_argument('-c', '--config', default=None, type=str,
                        help='path to JSON run config; defaults to demos/rpy_eval.config.json if present')
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='dataset root folder to scan recursively for images')
    parser.add_argument('--log-filename', default='rpy_evaluation.log', type=str,
                        help='summary log filename written to the dataset root')
    parser.add_argument('--frame-data-filename', default=FRAME_DATA_FILENAME, type=str,
                        help='ground truth JSON filename expected next to each source image')
    parser.add_argument('--html-report-filename', default='rpy_evaluation.html', type=str,
                        help='HTML visualization filename written to the dataset root')
    parser.add_argument('--visualize', default=True, type=str2bool,
                        help='whether to generate HTML deviation visualization report')
    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('-c', '--config', default=None)
    pre_args, remaining_argv = pre_parser.parse_known_args()

    config_path = resolve_config_path(pre_args.config)
    defaults = load_run_config(config_path) if config_path else {}

    parser = build_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    if config_path:
        print(f'loaded run config: {config_path}')

    return args


def main(args):
    dataset_root = normalize_input_path(args.inputpath)
    if not os.path.isdir(dataset_root):
        print(f'input path does not exist or is not a directory: {dataset_root}')
        return

    imagepath_list = collect_images_recursive(dataset_root)
    if not imagepath_list:
        print(f'no images found under: {dataset_root}')
        return

    print(f'found {len(imagepath_list)} image(s) under {dataset_root}')

    results = []
    skipped = []
    for imagepath in imagepath_list:
        rel_path = os.path.relpath(imagepath, dataset_root)
        try:
            results.append(
                evaluate_image(imagepath, dataset_root, args.frame_data_filename)
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append((rel_path, str(exc)))
            print(f'skipped {rel_path}: {exc}')

    log_path = os.path.join(dataset_root, args.log_filename)
    write_dataset_log(log_path, dataset_root, results, skipped)

    if args.visualize:
        report_path = os.path.join(dataset_root, args.html_report_filename)
        write_html_report(report_path, dataset_root, results, skipped)
        print(f'-- HTML report written to {report_path}')

    print(f'evaluated {len(results)} image(s), skipped {len(skipped)}')
    print(f'-- summary log written to {log_path}')


if __name__ == '__main__':
    main(parse_args())
