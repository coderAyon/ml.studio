"""Bounded association rules from explicit column=value transactions.

Each dataset row is one transaction. Missing selected cells contribute no item;
all rows remain in the support denominator. No numeric binning is inferred.
"""
from itertools import combinations
import math

import numpy as np
import pandas as pd


MAX_ROWS = 20_000
MAX_COLUMNS = 20
MAX_VALUES_PER_COLUMN = 50
MAX_ITEMS = 300
MAX_ITEMSETS = 3_000
MAX_CANDIDATES = 25_000
MAX_RULE_CANDIDATES = 30_000
MAX_TRANSACTION_MEMBERSHIPS = 1_000_000


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not .01 <= value <= 1:
        raise ValueError(f'{name} must be between 0.01 and 1.')
    return float(value)


def _item_value(value):
    """Keep typed category identities distinct, including integer 1 and text '1'."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return type(value).__name__, value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        if len(str(value)) > 500:
            raise ValueError('Association categories must be short labels (up to 500 characters). Choose category columns instead of free text.')
        return type(value).__name__, str(value)
    raise ValueError('Association rules need scalar category values. Convert nested/object cells to text first.')


def mine_rules(frame, config=None):
    """Return real Apriori itemsets and rules without modifying the frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Import data with at least two rows before finding association rules.')
    if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame.columns):
        raise ValueError('Use distinct text column names before finding association rules.')
    if not 2 <= len(frame) <= MAX_ROWS:
        raise ValueError(f'Association rules support 2–{MAX_ROWS:,} rows. Use a smaller extract.')
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError('Choose valid association-rule settings.')
    columns = config.get('columns', [])
    if (not isinstance(columns, list) or not 2 <= len(columns) <= MAX_COLUMNS
            or any(not isinstance(c, str) or c not in frame for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError(f'Choose 2–{MAX_COLUMNS} distinct existing category columns.')
    numeric_as_categories = config.get('numeric_as_categories', False)
    if type(numeric_as_categories) is not bool:
        raise ValueError('Choose whether numeric values should be treated as category labels.')
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
    if numeric and not numeric_as_categories:
        raise ValueError('Numeric columns require an explicit choice: treat their exact values as category labels, or select category columns instead. No automatic binning is applied.')
    support = _fraction(config.get('min_support', .1), 'Minimum support')
    confidence = _fraction(config.get('min_confidence', .6), 'Minimum confidence')
    max_length = config.get('max_length', 3)
    max_rules = config.get('max_rules', 100)
    if type(max_length) is not int or not 2 <= max_length <= 4:
        raise ValueError('Choose an itemset length from 2 to 4.')
    if type(max_rules) is not int or not 1 <= max_rules <= 300:
        raise ValueError('Show between 1 and 300 rules.')

    # Vertical transaction sets make support exact and avoid scanning every row
    # once for every candidate. IDs are local; no string expression is evaluated.
    item_ids, descriptions, tids = {}, [], {}
    for column in columns:
        seen=set()
        for position, value in enumerate(frame[column]):
            missing=pd.isna(value)
            if not isinstance(missing,(bool,np.bool_)):
                raise ValueError('Association rules need scalar category values. Convert nested/object cells to text first.')
            if missing:
                continue
            kind, label = _item_value(value)
            seen.add((kind,label))
            if len(seen)>MAX_VALUES_PER_COLUMN:
                raise ValueError(f'“{column}” has more than {MAX_VALUES_PER_COLUMN} values. Choose a category column, group values, or discretize it explicitly first.')
            identity = (column, kind, label)
            item = item_ids.get(identity)
            if item is None:
                if len(item_ids) >= MAX_ITEMS:
                    raise ValueError(f'Selected columns exceed {MAX_ITEMS} distinct items. Choose fewer columns or group rare values first.')
                item = len(item_ids)
                item_ids[identity] = item
                descriptions.append({'column': column, 'value': label, 'value_type': kind})
                tids[item] = set()
            tids[item].add(position)
    if not item_ids:
        raise ValueError('The selected columns contain only missing values. Choose columns with observed categories.')
    n = len(frame)
    minimum_count = max(1, math.ceil(support * n - 1e-12))
    frequent = {(item,): frozenset(rows) for item, rows in tids.items() if len(rows) >= minimum_count}
    counts = {items: len(rows) for items, rows in frequent.items()}
    candidate_count = 0
    for length in range(2, min(max_length, len(columns)) + 1):
        previous = sorted(frequent)
        previous_keys = set(previous)
        current_memberships = sum(len(rows) for rows in frequent.values())
        next_memberships = 0
        following = {}
        for i, left in enumerate(previous):
            for right in previous[i + 1:]:
                if left[:-1] != right[:-1]:
                    continue
                candidate = left + (right[-1],)
                # A row cannot contain two values from the same selected column.
                if len({descriptions[item]['column'] for item in candidate}) != length:
                    continue
                if not all(subset in previous_keys for subset in combinations(candidate, length - 1)):
                    continue
                candidate_count += 1
                if candidate_count > MAX_CANDIDATES:
                    raise ValueError('Too many association candidates. Increase minimum support, reduce itemset length, or select fewer columns.')
                rows = frequent[left].intersection(frequent[right])
                if len(rows) >= minimum_count:
                    next_memberships += len(rows)
                    if current_memberships + next_memberships > MAX_TRANSACTION_MEMBERSHIPS:
                        raise ValueError('Association patterns exceed the local memory budget. Choose fewer columns, raise support, or use fewer rows.')
                    following[candidate] = rows
                    counts[candidate] = len(rows)
                    if len(counts) > MAX_ITEMSETS:
                        raise ValueError('Too many frequent patterns. Increase minimum support or choose fewer columns.')
        frequent = following
        if not frequent:
            break

    def items_label(items):
        return ' AND '.join(f'{descriptions[item]["column"]} = {descriptions[item]["value"]!r}' for item in items)

    rules = []
    examined_rules = 0
    for items, count in counts.items():
        if len(items) < 2:
            continue
        for size in range(1, len(items)):
            for antecedent in combinations(items, size):
                examined_rules += 1
                if examined_rules > MAX_RULE_CANDIDATES:
                    raise ValueError('Too many possible rules. Increase support or reduce the maximum itemset length.')
                consequent = tuple(item for item in items if item not in antecedent)
                conf = count / counts[antecedent]
                if conf + 1e-12 < confidence:
                    continue
                support_a, support_b = counts[antecedent] / n, counts[consequent] / n
                rules.append({'antecedent': items_label(antecedent), 'consequent': items_label(consequent),
                    'antecedent_items': [descriptions[item] for item in antecedent],
                    'consequent_items': [descriptions[item] for item in consequent],
                    'count': count, 'support': count / n, 'confidence': conf, 'lift': conf / support_b,
                    'leverage': count / n - support_a * support_b,
                    'conviction': (1 - support_b) / (1 - conf) if conf < 1 else None})
    rules.sort(key=lambda rule: (-rule['lift'], -rule['confidence'], -rule['support'], rule['antecedent'], rule['consequent']))
    total_rules = len(rules)
    rules = rules[:max_rules]
    itemsets = [{'items': [descriptions[item] for item in items], 'label': items_label(items),
        'count': count, 'support': count / n} for items, count in sorted(counts.items(), key=lambda pair: (len(pair[0]), pair[0]))]
    table = {'columns': ['If these values occur', 'Then this value occurs', 'Rows', 'Support', 'Confidence', 'Lift', 'Leverage', 'Conviction'],
        'data': [[r['antecedent'], r['consequent'], r['count'], r['support'], r['confidence'], r['lift'], r['leverage'], r['conviction']] for r in rules],
        'index': list(range(len(rules)))}
    return {'rows': n, 'transaction_count': n, 'selected_columns': columns, 'item_count': len(item_ids),
        'minimum_count': minimum_count, 'frequent_itemsets': itemsets, 'total_rules': total_rules,
        'shown_rules': len(rules), 'truncated': total_rules > len(rules), 'rules': rules, 'table': table,
        'config': {'columns': columns, 'min_support': support, 'min_confidence': confidence,
            'max_length': max_length, 'max_rules': max_rules, 'numeric_as_categories': numeric_as_categories},
        'note': 'Each row is one transaction; each observed selected cell is a column=value item. Missing cells add no item, but every row stays in the support denominator. Rules describe co-occurrence in this dataset, not causation or validated predictions.',
        'metric_notes': {'support': 'Share of all rows containing both sides.',
            'confidence': 'Among rows containing the left side, share also containing the right side.',
            'lift': 'Confidence divided by the right side’s overall frequency. Above 1 means more co-occurrence than independence would suggest.',
            'conviction': 'Null means confidence is 1, so finite conviction is undefined.'},
        'limits': {'max_rows': MAX_ROWS, 'max_columns': MAX_COLUMNS, 'max_items': MAX_ITEMS,
            'max_frequent_itemsets': MAX_ITEMSETS, 'max_candidates': MAX_CANDIDATES,
            'max_transaction_memberships': MAX_TRANSACTION_MEMBERSHIPS}}
