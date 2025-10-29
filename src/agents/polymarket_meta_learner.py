"""
[EMOJI] Moon Dev's Polymarket Meta-Learner
Learns optimal agent weights from historical performance
Built with love by Moon Dev [EMOJI]
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from termcolor import cprint
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, log_loss, brier_score_loss
import joblib

# Add project root to path
import sys
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.config import POLYMARKET_DATA_DIR


class PolymarketMetaLearner:
    """
    Polymarket Meta-Learner - Agent Weight Optimizer

    Learns optimal weights for agent signals using:
    1. Short-horizon price movement prediction (profit target)
    2. Resolution outcome prediction (calibration target)

    Outputs:
    - calibration.json with per-segment weights
    - Model performance metrics
    - Feature importance rankings
    """

    def __init__(self):
        """Initialize Meta-Learner"""
        self.data_dir = Path(POLYMARKET_DATA_DIR) / 'meta_learning'
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.training_data_dir = Path(POLYMARKET_DATA_DIR) / 'training_data'

        # Model files
        self.calibration_file = self.data_dir / 'calibration.json'
        self.short_horizon_model_file = self.data_dir / 'short_horizon_model.pkl'
        self.resolution_model_file = self.data_dir / 'resolution_model.pkl'
        self.scaler_file = self.data_dir / 'scaler.pkl'

        # Models
        self.short_horizon_model = None
        self.resolution_model = None
        self.scaler = None

        # Load existing calibration
        self.calibration = self._load_calibration()

        cprint(f"\n[BRAIN] Polymarket Meta-Learner Initialized", "cyan", attrs=['bold'])
        cprint(f"[DATA] Calibration Version: {self.calibration.get('version', 0)}", "green")
        cprint(f"[CALENDAR] Last Updated: {self.calibration.get('updated_at', 'Never')}", "green")

    def _load_calibration(self) -> Dict:
        """Load existing calibration or create default"""
        if self.calibration_file.exists():
            with open(self.calibration_file, 'r') as f:
                calibration = json.load(f)
            cprint(f"[OK] Loaded existing calibration", "green")
            return calibration
        else:
            calibration = {
                'version': 0,
                'updated_at': None,
                'meta_model': 'ridge_v1',
                'segments': {}
            }
            cprint(f"[NOTE] Created default calibration", "yellow")
            return calibration

    def build_training_dataset(
        self,
        horizon_minutes: int = 120,
        min_samples_per_segment: int = 50
    ) -> pd.DataFrame:
        """
        Build training dataset from collected market snapshots

        Args:
            horizon_minutes: Forward horizon for short-term prediction
            min_samples_per_segment: Minimum samples required per segment

        Returns:
            Training dataset DataFrame
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[DATA] BUILDING TRAINING DATASET", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")
        cprint(f"[TIME]  Horizon: {horizon_minutes} minutes", "yellow")

        # Load collected data
        snapshots = self._load_snapshots()
        orderbook = self._load_orderbook()
        events = self._load_events()
        resolutions = self._load_resolutions()

        if snapshots.empty:
            cprint(f"[ERROR] No snapshots found. Run data collector first.", "red")
            return pd.DataFrame()

        cprint(f"[OK] Loaded {len(snapshots)} snapshots", "green")

        # Build feature matrix
        dataset = []

        # Group by market
        for market_id, market_snaps in snapshots.groupby('market_id'):
            market_snaps = market_snaps.sort_values('timestamp')

            # Skip if too few samples
            if len(market_snaps) < 10:
                continue

            # Get resolution if available
            if resolutions.empty:
                resolved = False
                resolved_outcome = None
            else:
                resolution = resolutions[resolutions['market_id'] == market_id]
                resolved = not resolution.empty
                resolved_outcome = resolution.iloc[0]['resolved_outcome'] if resolved else None

            # Create rows
            for i in range(len(market_snaps) - 1):
                current = market_snaps.iloc[i]
                current_time = current['timestamp']

                # Find forward snapshot (horizon_minutes ahead)
                future_snaps = market_snaps[
                    market_snaps['timestamp'] >= current_time + timedelta(minutes=horizon_minutes)
                ]

                if future_snaps.empty:
                    continue

                future = future_snaps.iloc[0]

                # Calculate forward price move (short-horizon target)
                delta_price_forward = future['mid_yes'] - current['mid_yes']

                # Get agent signals at current time
                current_orderbook = orderbook[
                    (orderbook['market_id'] == market_id) &
                    (orderbook['timestamp'] <= current_time)
                ].sort_values('timestamp')

                current_events = events[
                    (events['market_id'] == market_id) &
                    (events['timestamp'] <= current_time)
                ].sort_values('timestamp')

                # Build feature row
                row = {
                    # Market state
                    'market_id': market_id,
                    'timestamp': current_time,
                    'mid_yes': current['mid_yes'],
                    'mid_no': current['mid_no'],
                    'spread': current['spread'],
                    'liquidity': current['liquidity'],
                    'volume_24h': current['volume_24h'],
                    'volatility_lookback': current['volatility_lookback'],
                    'time_to_resolution_days': current['time_to_resolution_days'],
                    'market_type': current['market_type'],
                    'regime': current['regime'],

                    # Agent signals
                    'whale_strength': current_orderbook.iloc[-1]['whale_strength'] if not current_orderbook.empty else 0.0,
                    'book_imbalance': current_orderbook.iloc[-1]['book_imbalance'] if not current_orderbook.empty else 0.0,
                    'odds_velocity': current_orderbook.iloc[-1]['odds_velocity'] if not current_orderbook.empty else 0.0,

                    'catalyst_impact': current_events.iloc[-1]['catalyst_impact'] if not current_events.empty else 0.0,
                    'sentiment_score': current_events.iloc[-1]['sentiment_score'] if not current_events.empty else 0.0,
                    'match_score': current_events.iloc[-1]['match_score'] if not current_events.empty else 0.0,

                    # Anomaly placeholder (would come from anomaly agent)
                    'anomaly_flag': 0.0,
                    'anomaly_mag': 0.0,

                    # Targets
                    'delta_price_forward': delta_price_forward,
                    'resolved': resolved,
                    'resolved_outcome': resolved_outcome if resolved else np.nan
                }

                dataset.append(row)

        df = pd.DataFrame(dataset)

        cprint(f"[OK] Built dataset: {len(df)} rows", "green")

        # Show segment distribution
        if not df.empty:
            segment_counts = df.groupby(['market_type', 'regime']).size()
            cprint(f"\n[DATA] Segment Distribution:", "yellow")
            for (mtype, regime), count in segment_counts.items():
                cprint(f"   {mtype}:{regime} - {count} samples", "cyan")

        return df

    def train_meta_models(
        self,
        dataset: pd.DataFrame,
        alpha_ridge: float = 1.0,
        cv_splits: int = 5
    ) -> Dict:
        """
        Train meta-learning models

        Args:
            dataset: Training dataset
            alpha_ridge: Ridge regression regularization
            cv_splits: Cross-validation splits

        Returns:
            Training results dict
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[BRAIN] TRAINING META-MODELS", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        if dataset.empty:
            cprint(f"[ERROR] Empty dataset", "red")
            return {}

        # Define features
        feature_cols = [
            'whale_strength',
            'book_imbalance',
            'odds_velocity',
            'catalyst_impact',
            'sentiment_score',
            'match_score',
            'anomaly_flag',
            'anomaly_mag',
            'spread',
            'liquidity',
            'volatility_lookback',
            'time_to_resolution_days'
        ]

        # Add interaction terms
        dataset['whale_catalyst'] = dataset['whale_strength'] * dataset['catalyst_impact']
        dataset['sentiment_catalyst'] = dataset['sentiment_score'] * dataset['catalyst_impact']
        feature_cols.extend(['whale_catalyst', 'sentiment_catalyst'])

        X = dataset[feature_cols].fillna(0)
        y_short = dataset['delta_price_forward']

        # Scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # === MODEL 1: Short-Horizon Ridge Regression ===
        cprint(f"\n[TARGET] Training Short-Horizon Model (Ridge)...", "cyan")

        self.short_horizon_model = Ridge(alpha=alpha_ridge)

        # CV scores
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_short.iloc[train_idx], y_short.iloc[val_idx]

            self.short_horizon_model.fit(X_train, y_train)
            y_pred = self.short_horizon_model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)

        avg_mse = np.mean(cv_scores)
        cprint(f"   [OK] CV MSE: {avg_mse:.6f}", "green")

        # Refit on full data
        self.short_horizon_model.fit(X_scaled, y_short)

        # Extract coefficients (weights)
        coefficients = dict(zip(feature_cols, self.short_horizon_model.coef_))
        coefficients['intercept'] = self.short_horizon_model.intercept_

        cprint(f"\n[DATA] Feature Weights (Short-Horizon):", "yellow")
        sorted_coeffs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, weight in sorted_coeffs[:10]:
            cprint(f"   {feature:30s}: {weight:+.4f}", "cyan")

        # === MODEL 2: Resolution Logistic Regression ===
        cprint(f"\n[TARGET] Training Resolution Model (Logistic)...", "cyan")

        # Filter to resolved markets
        resolved_data = dataset[dataset['resolved'] == True].copy()

        if len(resolved_data) >= 50:
            X_res = resolved_data[feature_cols].fillna(0)
            X_res_scaled = self.scaler.transform(X_res)
            y_res = resolved_data['resolved_outcome']

            self.resolution_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

            # CV scores
            cv_scores_res = []
            for train_idx, val_idx in tscv.split(X_res_scaled):
                X_train, X_val = X_res_scaled[train_idx], X_res_scaled[val_idx]
                y_train, y_val = y_res.iloc[train_idx], y_res.iloc[val_idx]

                self.resolution_model.fit(X_train, y_train)
                y_pred = self.resolution_model.predict_proba(X_val)[:, 1]

                brier = brier_score_loss(y_val, y_pred)
                cv_scores_res.append(brier)

            avg_brier = np.mean(cv_scores_res)
            cprint(f"   [OK] CV Brier Score: {avg_brier:.4f}", "green")

            # Refit
            self.resolution_model.fit(X_res_scaled, y_res)

            # Extract coefficients
            res_coefficients = dict(zip(feature_cols, self.resolution_model.coef_[0]))

            cprint(f"\n[DATA] Feature Weights (Resolution):", "yellow")
            sorted_res = sorted(res_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, weight in sorted_res[:10]:
                cprint(f"   {feature:30s}: {weight:+.4f}", "cyan")
        else:
            cprint(f"   [WARN]  Not enough resolved markets ({len(resolved_data)}). Skipping.", "yellow")
            self.resolution_model = None

        # Save models
        self._save_models()

        results = {
            'short_horizon_mse': avg_mse,
            'short_horizon_coefficients': coefficients,
            'resolution_brier': avg_brier if self.resolution_model else None,
            'resolution_coefficients': res_coefficients if self.resolution_model else None,
            'feature_names': feature_cols
        }

        return results

    def train_per_segment(
        self,
        dataset: pd.DataFrame
    ) -> Dict:
        """
        Train separate models per market type × regime segment

        Returns:
            Segment-specific calibration
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[SEGMENT] TRAINING PER-SEGMENT MODELS", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        segments = {}

        # Group by market_type × regime
        for (market_type, regime), segment_data in dataset.groupby(['market_type', 'regime']):
            segment_key = f"{market_type}:{regime}"

            cprint(f"\n[DATA] Training segment: {segment_key} ({len(segment_data)} samples)", "cyan")

            if len(segment_data) < 30:
                cprint(f"   [WARN]  Too few samples. Skipping.", "yellow")
                continue

            # Train on this segment
            feature_cols = [
                'whale_strength', 'book_imbalance', 'odds_velocity',
                'catalyst_impact', 'sentiment_score', 'match_score',
                'anomaly_flag', 'anomaly_mag',
                'spread', 'liquidity', 'volatility_lookback'
            ]

            X = segment_data[feature_cols].fillna(0)
            y_short = segment_data['delta_price_forward']

            # Scale
            scaler_seg = StandardScaler()
            X_scaled = scaler_seg.fit_transform(X)

            # Ridge
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_short)

            # Get coefficients
            weights = dict(zip(feature_cols, model.coef_))
            weights['intercept'] = model.intercept_

            # Calculate sigma (std of residuals)
            y_pred = model.predict(X_scaled)
            residuals = y_short - y_pred
            sigma_short = np.std(residuals)

            # Kelly multiplier based on regime
            kelly_mult = {
                'information': 1.0,
                'emotion': 1.5,
                'illiquid': 0.5
            }.get(regime, 1.0)

            segments[segment_key] = {
                'weights': weights,
                'sigma_short': sigma_short,
                'kelly_multiplier': kelly_mult,
                'samples': len(segment_data)
            }

            cprint(f"   [OK] Trained with sigma={sigma_short:.4f}, Kelly={kelly_mult}", "green")

        return segments

    def generate_calibration_json(
        self,
        segments: Dict
    ):
        """
        Generate calibration.json file

        Args:
            segments: Segment-specific weights
        """
        cprint(f"\n{'='*80}", "cyan")
        cprint(f"[FILE] GENERATING CALIBRATION.JSON", "cyan", attrs=['bold'])
        cprint(f"{'='*80}", "cyan")

        new_version = self.calibration.get('version', 0) + 1

        calibration = {
            'meta_model': 'ridge_v1',
            'version': new_version,
            'updated_at': datetime.now().isoformat(),
            'segments': segments
        }

        # Save
        with open(self.calibration_file, 'w') as f:
            json.dump(calibration, f, indent=2)

        cprint(f"[OK] Calibration v{new_version} saved to {self.calibration_file.name}", "green")

        # Update instance
        self.calibration = calibration

    def _load_snapshots(self) -> pd.DataFrame:
        """Load market snapshots"""
        file_path = self.training_data_dir / 'market_snapshots.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()

    def _load_orderbook(self) -> pd.DataFrame:
        """Load orderbook snapshots"""
        file_path = self.training_data_dir / 'orderbook_snapshots.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()

    def _load_events(self) -> pd.DataFrame:
        """Load events"""
        file_path = self.training_data_dir / 'event_snapshots.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()

    def _load_resolutions(self) -> pd.DataFrame:
        """Load market resolutions"""
        file_path = self.training_data_dir / 'market_resolutions.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['resolution_time'] = pd.to_datetime(df['resolution_time'])
            return df
        return pd.DataFrame()

    def _save_models(self):
        """Save trained models"""
        if self.short_horizon_model:
            joblib.dump(self.short_horizon_model, self.short_horizon_model_file)
            cprint(f"[SAVE] Saved short-horizon model", "blue")

        if self.resolution_model:
            joblib.dump(self.resolution_model, self.resolution_model_file)
            cprint(f"[SAVE] Saved resolution model", "blue")

        if self.scaler:
            joblib.dump(self.scaler, self.scaler_file)
            cprint(f"[SAVE] Saved scaler", "blue")

    def run_training_pipeline(self):
        """
        Run complete training pipeline

        1. Build dataset
        2. Train global models
        3. Train per-segment models
        4. Generate calibration.json
        """
        cprint(f"\n{'='*80}", "magenta")
        cprint(f"[LEARN] META-LEARNING TRAINING PIPELINE", "magenta", attrs=['bold'])
        cprint(f"{'='*80}\n", "magenta")

        # Step 1: Build dataset
        dataset = self.build_training_dataset(horizon_minutes=120)

        if dataset.empty:
            cprint(f"[ERROR] Cannot train without data", "red")
            return

        # Step 2: Train global models
        global_results = self.train_meta_models(dataset)

        # Step 3: Train per-segment models
        segments = self.train_per_segment(dataset)

        # Step 4: Generate calibration
        self.generate_calibration_json(segments)

        cprint(f"\n{'='*80}", "green")
        cprint(f"[OK] TRAINING PIPELINE COMPLETE", "green", attrs=['bold'])
        cprint(f"{'='*80}\n", "green")


def main():
    """Run meta-learner training"""

    learner = PolymarketMetaLearner()

    # Run training pipeline
    learner.run_training_pipeline()


if __name__ == "__main__":
    main()
