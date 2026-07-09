"""BTC cycle metadata and local segment helpers for residual analysis."""

import json
from datetime import datetime

import numpy as np

from .constants import DAYS_PER_YEAR


LOC_SEGMENTS = [('H2-H3', 2, 3), ('H3-H4', 3, 4), ('H4+', 4, None)]


def load_halvings(cycles_json):
    with open(cycles_json, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)
    rows = []
    for row in payload['halvings']:
        date_text = row.get('date')
        rows.append({
            'nr': int(row['nr']),
            'name': row['name'],
            'index_abs': int(row['index_abs']),
            'date_text': date_text,
            'date': None if not date_text else datetime.strptime(date_text, '%d.%m.%Y'),
        })
    return rows


def build_loc_segment_rows(
    days_vecs,
    cycles_json,
    main_period_years=None,
    sub_period_years=None,
    support_mask=None,
):
    days_vecs = np.asarray(days_vecs, dtype=float)
    halvings = {row['nr']: row for row in load_halvings(cycles_json)}
    rows = []
    for label, start_nr, end_nr in LOC_SEGMENTS:
        start = halvings[start_nr]
        end = None if end_nr is None else halvings[end_nr]
        start_day = float(start['index_abs'])
        end_day = np.inf if end is None else float(end['index_abs'])
        mask = (days_vecs >= start_day) if end is None else ((days_vecs >= start_day) & (days_vecs < end_day))
        n_vec = int(np.sum(mask))
        if end is None:
            span_days = float(max(days_vecs[-1] - start_day, 0.0))
        else:
            span_days = float(max(end_day - start_day, 0.0))
        row = {
            'label': label,
            'start_day': start_day,
            'end_day': end_day,
            'start_date_text': start['date_text'],
            'end_date_text': None if end is None else end['date_text'],
            'n_vec': n_vec,
            'span_days': span_days,
            'cycles_main': None if not main_period_years else float(span_days / (main_period_years * DAYS_PER_YEAR)),
            'cycles_sub': None if not sub_period_years else float(span_days / (sub_period_years * DAYS_PER_YEAR)),
            'support_fraction': None,
        }
        if support_mask is not None and n_vec > 0:
            row['support_fraction'] = float(np.mean(np.asarray(support_mask, dtype=bool)[mask]))
        rows.append(row)
    return rows
