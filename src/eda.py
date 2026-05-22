"""
eda.py
======
Student Performance Prediction — Exploratory Data Analysis Module
Decodelabs Data Science Internship — Task 3


Usage:
    from src.eda import (
        grade_statistics, correlation_analysis,
        group_analysis, outlier_report, eda_summary
    )
"""

import pandas as pd
import numpy as np

#  Grade statistics 

def grade_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive statistics for G1, G2, G3.
    Returns a dict so the notebook can display however it wants.
    """
    stats = {}
    for col in ['G1', 'G2', 'G3']:
        stats[col] = {
            'mean'    : round(df[col].mean(), 2),
            'median'  : round(df[col].median(), 2),
            'std'     : round(df[col].std(), 2),
            'min'     : int(df[col].min()),
            'max'     : int(df[col].max()),
            'skew'    : round(df[col].skew(), 3),
            'pass_pct': round((df[col] >= 10).mean() * 100, 1),
        }
    return stats


#  Correlation analysis 

def correlation_analysis(df: pd.DataFrame, target: str = 'G3') -> pd.Series:
    """
    Return correlation of all numeric columns with the target,
    sorted strongest to weakest (absolute value).
    """
    corr = df.corr(numeric_only=True)[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


#  Group analysis 

def group_analysis(df: pd.DataFrame, group_col: str,
                   target: str = 'G3') -> pd.DataFrame:
    """
    Compute mean grade and pass rate for each value of group_col.
    """
    result = df.groupby(group_col).agg(
        count      = (target, 'count'),
        mean_grade = (target, 'mean'),
        pass_rate  = ('pass_fail', 'mean'),
    ).round(3)
    result['mean_grade'] = result['mean_grade'].round(2)
    result['pass_rate']  = (result['pass_rate'] * 100).round(1)
    return result

#  Outlier report 

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier detection for all numerical columns.
    Returns a summary DataFrame.
    """
    num_cols = df.select_dtypes(include='number').columns
    rows = []
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower   = Q1 - 1.5 * IQR
        upper   = Q3 + 1.5 * IQR
        n_out   = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            'column'       : col,
            'Q1'           : round(Q1, 2),
            'Q3'           : round(Q3, 2),
            'IQR'          : round(IQR, 2),
            'lower_bound'  : round(lower, 2),
            'upper_bound'  : round(upper, 2),
            'n_outliers'   : n_out,
            'outlier_pct'  : round(n_out / len(df) * 100, 1),
        })
    return pd.DataFrame(rows).set_index('column')

#  Grade progression 

def grade_progression(df: pd.DataFrame) -> dict:
    """Analyse how grades changed from G1 -> G2 -> G3."""
    return {
        'improved' : int((df['G3'] > df['G1']).sum()),
        'dropped'  : int((df['G3'] < df['G1']).sum()),
        'same'     : int((df['G3'] == df['G1']).sum()),
        'G1_mean'  : round(df['G1'].mean(), 2),
        'G2_mean'  : round(df['G2'].mean(), 2),
        'G3_mean'  : round(df['G3'].mean(), 2),
    }


#  EDA summary 

def eda_summary(df: pd.DataFrame) -> None:
    """Print the full Task 3 written summary."""
    corr = correlation_analysis(df)
    prog = grade_progression(df)

    print("=" * 62)
    print("  TASK 3 SUMMARY - Exploratory Data Analysis")
    print("=" * 62)
    print(f"""
  Dataset
    Rows         : {df.shape[0]}
    Columns      : {df.shape[1]}
    Pass rate    : {df['pass_fail'].mean()*100:.1f}%
    Fail rate    : {(1-df['pass_fail'].mean())*100:.1f}%

  Grade statistics
    G1 mean      : {prog['G1_mean']}   (first term)
    G2 mean      : {prog['G2_mean']}   (second term)
    G3 mean      : {prog['G3_mean']}   (final — our target)
    Grade trend  : slightly declining each term

  Grade progression G1 → G3
    Improved     : {prog['improved']} students
    Dropped      : {prog['dropped']} students
    Stayed same  : {prog['same']} students

  Top 5 positive correlations with G3
    {corr[corr > 0].head(5).to_string()}

  Top 5 negative correlations with G3
    {corr[corr < 0].head(5).to_string()}

  Key findings
    1. G1 and G2 are by far the strongest predictors of G3
       (r=0.80, r=0.90). Early performance matters most.
    2. Past failures strongly hurt final grades.
       0 failures → avg G3=11.25 vs 3 failures → avg G3=5.69
    3. Students who want higher education pass at 69% vs 35%
       for those who don't — motivation is measurable.
    4. Alcohol consumption (Dalc, Walc) negatively correlates
       with grades but the effect is moderate, not dominant.
    5. Absences is right-skewed — most students miss very few
       days, but a small group misses many.
    6. Failures column is highly skewed (skew=2.39) —
       most students have 0 failures.
""")
    print("=" * 62)
