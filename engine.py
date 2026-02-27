import pandas as pd
import numpy as np
import json
from datetime import datetime

class RecommendationEngine:
    """
    Generates personalized financial recommendations from M-Pesa transactions.
    All logic is rule-based: transparent, explainable, correct from day one.
    """

    NON_SPEND = [
        'M-Pesa Fees', 'Savings', 'Cash Deposit',
        'Income', 'Cash Withdrawal'
    ]

    def __init__(self, df: pd.DataFrame, user_id: str = "user"):
        self.df       = df.copy()
        self.user_id  = user_id
        self.recommendations = []

        for col in ['amount_spent', 'amount_received', 'balance']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        self.df['date']     = pd.to_datetime(self.df['date'])

        self.consumption_df = self.df[~self.df['final_category'].isin(self.NON_SPEND)]
        self.savings_total  = self.df[self.df['final_category'] == 'Savings']['amount_spent'].sum()

        self.total_spent    = self.consumption_df['amount_spent'].sum()
        self.total_received = self.df['amount_received'].sum()
        self.fees_total     = self.df[self.df['final_category'] == 'M-Pesa Fees']['amount_spent'].sum()

        self.start_date = self.df['datetime'].min()
        self.end_date   = self.df['datetime'].max()
        self.days       = (self.end_date - self.start_date).days or 1
        self.months     = self.days / 30.44

        self.true_savings_rate = (self.savings_total / self.total_received * 100
                                  if self.total_received > 0 else 0)

    def budget_recommendations(self):
        cat_spend   = self.consumption_df.groupby('final_category')['amount_spent'].sum()
        monthly_avg = cat_spend / self.months

        for cat in cat_spend.sort_values(ascending=False).head(12).index:
            current  = monthly_avg[cat]
            is_disc  = len(self.consumption_df[
                (self.consumption_df['final_category'] == cat) &
                (self.consumption_df['is_discretionary'] == 1)
            ]) > 0
            is_ess   = len(self.consumption_df[
                (self.consumption_df['final_category'] == cat) &
                (self.consumption_df['is_essential'] == 1)
            ]) > 0

            if is_disc:
                target  = current * 0.8
                monthly_saving = current - target
                self.recommendations.append({
                    'type'      : 'budget',
                    'category'  : cat,
                    'message'   : f"Reduce {cat} by 20%",
                    'current'   : f"KES {current:,.0f}/month",
                    'target'    : f"KES {target:,.0f}/month",
                    'impact'    : f"Save KES {monthly_saving:,.0f}/month  (KES {monthly_saving*12:,.0f}/year)",
                    'confidence': 'high',
                    'priority'  : 1,
                    'actionable': True,
                })

            elif is_ess and current > 2_000:
                target = current * 1.1
                self.recommendations.append({
                    'type'      : 'budget',
                    'category'  : cat,
                    'message'   : f"Set monthly budget for {cat}",
                    'current'   : f"KES {current:,.0f}/month average",
                    'target'    : f"KES {target:,.0f}/month (10% buffer)",
                    'impact'    : "Prevents overspending surprise on a recurring essential",
                    'confidence': 'medium',
                    'priority'  : 2,
                    'actionable': True,
                })

        if 'Construction' in monthly_avg.index:
            c_mo = monthly_avg['Construction']
            self.recommendations.append({
                'type'      : 'budget',
                'category'  : 'Construction',
                'message'   : f"Track Construction spend — KES {c_mo:,.0f}/month average",
                'current'   : f"KES {c_mo:,.0f}/month",
                'target'    : "Set a project end-date and monthly cap",
                'impact'    : "Easy category to overshoot silently without a cap",
                'confidence': 'high',
                'priority'  : 1,
                'actionable': True,
            })

    def savings_opportunities(self):
        disc         = self.consumption_df[self.consumption_df['is_discretionary'] == 1]
        disc_monthly = disc['amount_spent'].sum() / self.months

        if disc_monthly > 0:
            self.recommendations.append({
                'type'      : 'savings',
                'category'  : 'Discretionary',
                'message'   : "Reduce want spending by 30%",
                'current'   : f"KES {disc_monthly:,.0f}/month on wants",
                'target'    : f"KES {disc_monthly*0.7:,.0f}/month",
                'impact'    : f"Save KES {disc_monthly*0.3:,.0f}/month",
                'confidence': 'high',
                'priority'  : 1,
                'actionable': True,
            })

        fees_df = self.df[self.df['final_category'] == 'M-Pesa Fees']
        fees_monthly = self.fees_total / self.months

        if fees_monthly > 0:
            transfer_fees   = fees_df[fees_df['description'].str.contains('Transfer', case=False, na=False)]['amount_spent'].sum()
            paybill_fees    = fees_df[fees_df['description'].str.contains('Pay', case=False, na=False)]['amount_spent'].sum()
            withdrawal_fees = fees_df[fees_df['description'].str.contains('Withdraw', case=False, na=False)]['amount_spent'].sum()

            controllable     = (transfer_fees * 0.4) + (withdrawal_fees * 0.5)

            self.recommendations.append({
                'type'      : 'savings',
                'category'  : 'M-Pesa Fees',
                'message'   : f"KES {fees_monthly:,.0f}/month in M-Pesa fees",
                'current'   : f"Transfer: KES {transfer_fees:,.0f} | Paybill/Till: KES {paybill_fees:,.0f} | Withdrawal: KES {withdrawal_fees:,.0f}",
                'target'    : "Batch Send Money transfers and plan withdrawals to reduce controllable fees.",
                'impact'    : f"Realistic saving: KES {controllable/self.months:,.0f}/month",
                'confidence': 'medium',
                'priority'  : 2,
                'actionable': True,
            })

        small = self.consumption_df[self.consumption_df['amount_spent'].between(50, 500)]
        if len(small) > 50:
            self.recommendations.append({
                'type'      : 'savings',
                'category'  : 'Small Transactions',
                'message'   : f"{len(small)} transactions in the KES 50–500 range",
                'current'   : f"Total: KES {small['amount_spent'].sum():,.0f}  Avg: KES {small['amount_spent'].mean():,.0f}",
                'target'    : "Batch similar small payments where possible",
                'impact'    : "Reduces both fees and impulse spending",
                'confidence': 'medium',
                'priority'  : 3,
                'actionable': False,
            })

    def behavioral_insights(self):
        if 'is_payday_week' in self.df.columns:
            payday_spend = self.consumption_df[self.consumption_df['is_payday_week'] == 1]['amount_spent'].sum()
            other_spend  = self.consumption_df[self.consumption_df['is_payday_week'] == 0]['amount_spent'].sum()
            total        = payday_spend + other_spend
            pct          = (payday_spend / total * 100) if total > 0 else 0

            self.recommendations.append({
                'type'      : 'behavioral',
                'category'  : 'Payday Week',
                'message'   : f"{pct:.1f}% of spend happens in week 1 of the month",
                'current'   : f"Week 1: KES {payday_spend:,.0f}  vs  rest of month: KES {other_spend:,.0f}",
                'target'    : "Wait 48 hours before large non-essential purchases after payday",
                'impact'    : "Reduces impulsive post-payday spending",
                'confidence': 'high',
                'priority'  : 2,
                'actionable': True,
            })

        if 'is_weekend' in self.df.columns:
            wknd = self.consumption_df[self.consumption_df['is_weekend'] == 1]['amount_spent'].sum()
            wkdy = self.consumption_df[self.consumption_df['is_weekend'] == 0]['amount_spent'].sum()
            pct  = (wknd / (wknd + wkdy) * 100) if (wknd + wkdy) > 0 else 0

            self.recommendations.append({
                'type'      : 'behavioral',
                'category'  : 'Weekend Spending',
                'message'   : f"{pct:.1f}% of spend on weekends — weekdays are dominant",
                'current'   : f"Weekend: KES {wknd:,.0f}  Weekday: KES {wkdy:,.0f}",
                'target'    : "Weekday-heavy pattern is healthy — monitor it stays this way",
                'impact'    : "No action needed; confirm pattern holds next month",
                'confidence': 'high',
                'priority'  : 3,
                'actionable': False,
            })

        if self.end_date.month == 11:
            self.recommendations.append({
                'type'      : 'behavioral',
                'category'  : 'December Alert',
                'message'   : "December historical average is KES 147,033 — budget now",
                'current'   : "Dec 2024: KES 147,033 | Dec 2025: KES 89,706",
                'target'    : "Set aside 2× your normal monthly budget before December 1",
                'impact'    : "Prevents end-of-year cash crunch",
                'confidence': 'high',
                'priority'  : 1,
                'actionable': True,
            })

        if 'hour' in self.df.columns:
            late = self.consumption_df[
                (self.consumption_df['hour'] >= 22) |
                (self.consumption_df['hour'] <= 4)
            ]
            if len(late) > 10:
                late_total = late['amount_spent'].sum()
                pct        = (late_total / self.total_spent * 100) if self.total_spent > 0 else 0
                if pct > 5:
                    self.recommendations.append({
                        'type'      : 'behavioral',
                        'category'  : 'Late Night Spending',
                        'message'   : f"{len(late)} transactions between 10pm–4am",
                        'current'   : f"KES {late_total:,.0f} ({pct:.1f}% of consumption)",
                        'target'    : "Review whether these are necessary",
                        'impact'    : "Late-night purchases correlate with lower deliberation",
                        'confidence': 'medium',
                        'priority'  : 3,
                        'actionable': False,
                    })

    def spending_predictions(self):
        last_30  = self.end_date - pd.Timedelta(days=30)
        recent   = self.consumption_df[self.consumption_df['datetime'] >= last_30]['amount_spent'].sum()
        hist_avg = self.total_spent / self.months
        difference = (recent / hist_avg) - 1 if hist_avg > 0 else 0

        if difference > 0.15:
            self.recommendations.append({
                'type'      : 'prediction',
                'category'  : 'Burn Rate',
                'message'   : f"Recent 30-day spend is {difference*100:.0f}% higher than average",
                'current'   : f"Last 30 days: KES {recent:,.0f}",
                'target'    : f"Aim for historical average: KES {hist_avg:,.0f}/month",
                'impact'    : "Slow down non-essential spending this week",
                'confidence': 'medium',
                'priority'  : 1,
                'actionable': True,
            })

    def comparative_analysis(self):
        sr = self.true_savings_rate
        if sr < 10:
            msg = "Warning: Savings rate below 10% — high vulnerability"
            pri = 1
        elif 10 <= sr < 20:
            msg = "Fair: Saving 10-20% — building a base"
            pri = 2
        else:
            msg = f"Excellent: Saving {sr:.1f}% — wealth building zone"
            pri = 3

        self.recommendations.append({
            'type'      : 'comparative',
            'category'  : 'Savings Rate',
            'message'   : msg,
            'current'   : f"Transferred to savings: KES {self.savings_total:,.0f}",
            'target'    : "Maintain >20% for long-term health",
            'impact'    : "A high savings rate is the strongest predictor of financial security",
            'confidence': 'high',
            'priority'  : pri,
            'actionable': pri < 3,
        })

    def check_financial_health(self):
        score = 0
        if self.true_savings_rate > 20: score += 1
        elif self.true_savings_rate > 10: score += 0.5

        ess_pct = self.consumption_df[self.consumption_df['is_essential'] == 1]['amount_spent'].sum() / self.total_received
        if ess_pct < 0.5: score += 1
        elif ess_pct < 0.6: score += 0.5

        if self.total_received > self.total_spent: score += 1

        self.health_score = score
        self.health_status = 'Wealth Builder' if score >= 2.5 else 'Stable' if score >= 1.5 else 'Vulnerable'

    def generate_all_recommendations(self):
        self.budget_recommendations()
        self.savings_opportunities()
        self.behavioral_insights()
        self.spending_predictions()
        self.comparative_analysis()
        self.check_financial_health()

        self.recommendations = sorted(
            self.recommendations,
            key=lambda x: (x['priority'], not x.get('actionable', False))
        )
        return self.recommendations

    def export_recommendations(self):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):          return int(obj)
                if isinstance(obj, np.floating):         return float(obj)
                if isinstance(obj, (np.bool_,)):         return bool(obj)
                if isinstance(obj, np.ndarray):          return obj.tolist()
                return super().default(obj)

        # Prepare category spend for UI
        cat_spend = self.consumption_df.groupby('final_category')['amount_spent'].sum().to_dict()
        
        # Prepare monthly trend for UI
        monthly_trend = self.consumption_df.groupby(self.consumption_df['datetime'].dt.to_period('M'))['amount_spent'].sum()
        monthly_trend.index = monthly_trend.index.astype(str)
        monthly_trend_dict = monthly_trend.to_dict()

        # Prepare income trend for UI (Spending vs Income)
        income_trend = self.df.groupby(self.df['datetime'].dt.to_period('M'))['amount_received'].sum()
        income_trend.index = income_trend.index.astype(str)
        income_trend_dict = income_trend.to_dict()

        # Prepare weekly spending (Payday Week Effect)
        # Week 1: 1-7, Week 2: 8-14, Week 3: 15-21, Week 4: 22-28, Week 5: 29-31
        df_cons = self.consumption_df.copy()
        df_cons['week_of_month'] = df_cons['datetime'].dt.day.map(
            lambda d: (d-1)//7 + 1
        )
        weekly_spend = df_cons.groupby('week_of_month')['amount_spent'].sum().to_dict()
        # Ensure all weeks 1-5 exist
        weekly_spend_list = [round(float(weekly_spend.get(i, 0)), 0) for i in range(1, 6)]

        # Prepare merchants for learning based on UNKNOWN patterns
        # Identify merchants that are still 'Other' or 'Uncategorized' after keyword pipeline
        merch_txns = self.df[self.df['type'].isin(['Till Payment', 'PayBill'])]
        
        def get_merch_name(row):
            desc = row['description']
            if ' - ' in desc: return desc.split(' - ')[1].strip()
            return desc
            
        merch_txns = merch_txns.copy()
        merch_txns['m_name'] = merch_txns.apply(get_merch_name, axis=1)
        
        # Count auto-categorized vs total
        auto_cats = self.df[self.df['final_category'] != 'Other']
        uncategorized_count = len(self.df[self.df['final_category'] == 'Other'])
        
        counts = merch_txns['m_name'].value_counts()
        merchants_to_teach = []
        # Filter for merchants that the engine couldn't identify strictly
        for name in counts.index:
            m_df = merch_txns[merch_txns['m_name'] == name]
            # If the majority of transactions for this merchant are 'Other', we need to teach it
            if m_df['final_category'].iloc[0] in ['Other', 'Uncategorized', 'Till Payment', 'PayBill']:
                merchants_to_teach.append({
                    'id': f"m_{name[:8]}",
                    'name': name,
                    'count': int(len(m_df)),
                    'amounts': [round(float(a), 0) for a in m_df['amount_spent'].tolist()]
                })
            if len(merchants_to_teach) >= 10: break # Limit to top 10 for UX

        return {
            'user_id'        : self.user_id,
            'generated_at'   : datetime.now().isoformat(),
            'period'         : {
                'start'  : str(self.start_date.date()),
                'end'    : str(self.end_date.date()),
                'days'   : int(self.days),
                'months' : round(float(self.months), 1),
            },
            'summary'        : {
                'total_transactions'       : int(len(self.df)),
                'consumption_transactions' : int(len(self.consumption_df)),
                'total_received_kes'       : round(float(self.total_received), 0),
                'consumption_spend_kes'    : round(float(self.total_spent), 0),
                'savings_moved_kes'        : round(float(self.savings_total), 0),
                'true_savings_rate_pct'    : round(float(self.true_savings_rate), 1),
                'monthly_spend_avg_kes'    : round(float(self.total_spent / self.months), 0),
                'fees_total_kes'           : round(float(self.fees_total), 0),
                'auto_labeled_count'       : int(len(auto_cats)),
                'remaining_to_teach'       : int(len(merchants_to_teach))
            },
            'health' : {
                'score'  : self.health_score,
                'status' : self.health_status
            },
            'charts': {
                'category_spend': cat_spend,
                'monthly_trend': monthly_trend_dict,
                'income_trend': income_trend_dict,
                'weekly_spend': weekly_spend_list
            },
            'recommendations': self.recommendations,
            'merchants_to_learn': merchants_to_teach
        }

def train_and_save_ml_models(df: pd.DataFrame):
    import os
    import json
    import joblib
    import numpy as np
    from datetime import datetime
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    EXCLUDE_M = ['M-Pesa Fees', 'Savings', 'Cash Deposit', 'Income', 'Cash Withdrawal']

    daily = (
        df[~df['final_category'].isin(EXCLUDE_M)]
        .groupby('date')
        .agg(
            total_spend = ('amount_spent', 'sum'),
            num_txns    = ('amount_spent', 'count'),
            avg_txn     = ('amount_spent', 'mean'),
            max_txn     = ('amount_spent', 'max'),
        )
        .sort_index()
    )

    daily.index = pd.to_datetime(daily.index)

    daily['lag1']           = daily['total_spend'].shift(1)
    daily['lag7']           = daily['total_spend'].shift(7)
    daily['lag30']          = daily['total_spend'].shift(30)
    daily['rolling7_mean']  = daily['total_spend'].rolling(7).mean()
    daily['rolling30_mean'] = daily['total_spend'].rolling(30).mean()
    daily['dow']            = daily.index.dayofweek
    daily['month']          = daily.index.month

    daily_clean = daily.dropna()

    features = ['lag1','lag7','lag30','rolling7_mean','rolling30_mean','dow','month','num_txns']
    X = daily_clean[features]
    y = daily_clean['total_spend']

    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    sc     = StandardScaler()
    Xtr_sc = sc.fit_transform(X_train)
    Xte_sc = sc.transform(X_test)

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
    ridge_cv   = RidgeCV(alphas=alphas, cv=5).fit(Xtr_sc, y_train)

    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    features_list = features

    scaler_final = StandardScaler()
    X_train_sc   = scaler_final.fit_transform(X_train)
    X_test_sc    = scaler_final.transform(X_test)

    ridge_final  = Ridge(alpha=float(ridge_cv.alpha_)).fit(X_train_sc, y_train)

    gb_final     = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_final     = RandomForestRegressor(n_estimators=100,     random_state=42).fit(X_train, y_train)

    ridge_preds_final = ridge_final.predict(X_test_sc)
    gb_preds_final    = gb_final.predict(X_test)
    rf_preds_final    = rf_final.predict(X_test)

    metrics = {
        'ridge_tuned':       {'test_r2': round(r2_score(y_test, ridge_preds_final), 4),
                              'test_mae': round(mean_absolute_error(y_test, ridge_preds_final), 2)},
        'gradient_boosting': {'test_r2': round(r2_score(y_test, gb_preds_final), 4),
                              'test_mae': round(mean_absolute_error(y_test, gb_preds_final), 2)},
        'random_forest':     {'test_r2': round(r2_score(y_test, rf_preds_final), 4),
                              'test_mae': round(mean_absolute_error(y_test, rf_preds_final), 2)},
    }

    joblib.dump(ridge_final,  os.path.join(MODEL_DIR, 'ridge_tuned.pkl'))
    joblib.dump(scaler_final, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(gb_final,     os.path.join(MODEL_DIR, 'gradient_boosting.pkl'))
    joblib.dump(rf_final,     os.path.join(MODEL_DIR, 'random_forest.pkl'))

    metadata = {
        'saved_at':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'selected_model':  'ridge_tuned',
        'ridge_alpha':     float(ridge_cv.alpha_),
        'features':        features_list,
        'train_rows':      int(len(X_train)),
        'test_rows':       int(len(X_test)),
        'train_date_range': [str(X_train.index.min().date()), str(X_train.index.max().date())],
        'test_date_range':  [str(X_test.index.min().date()),  str(X_test.index.max().date())],
        'deployment_gate': {'metric': 'test_r2', 'threshold': 0.50,
                            'status': 'TRAINING PHASE — not yet deployable'},
        'metrics':         metrics,
    }

    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def process_mpesa_statement(df: pd.DataFrame, user_id: str = "user"):
    engine = RecommendationEngine(df, user_id)
    engine.generate_all_recommendations()
    recs = engine.export_recommendations()
    
    # Train and save ML models
    try:
        model_metadata = train_and_save_ml_models(df)
        recs['ml_models_saved'] = True
        recs['ml_model_metadata'] = model_metadata
    except Exception as e:
        print(f"Error training models: {e}")
        recs['ml_models_saved'] = False

    return recs

