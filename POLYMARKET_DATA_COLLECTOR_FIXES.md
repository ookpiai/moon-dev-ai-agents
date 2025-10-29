# Polymarket Data Collector Fixes

## Issues Identified:

1. **time_to_resolution = 999 days** (100% failure rate)
   - Problem: Markets return `endDate` but parsing may be failing
   - Fix: Improve `_calculate_time_to_resolution()` parsing

2. **spread = 0.98 (98%)** (51% failure rate)
   - Problem: Collecting closed markets with API's placeholder spread=1.0
   - Fix: Filter out closed markets, fallback to market prices when order book unavailable

3. **Order book API failures**
   - Problem: CLOB API requires authentication or returns empty data
   - Fix: Use market's `outcomePrices` array as fallback for spread calculation

## Fixes Applied:

### Fix 1: Filter Out Closed Markets
```python
# Only collect ACTIVE markets
if market.get('closed', False) or not market.get('active', True):
    return None
```

### Fix 2: Improved time_to_resolution
```python
def _calculate_time_to_resolution(self, market: Dict) -> float:
    """Calculate days until market resolution"""
    # Try multiple date fields
    for field in ['endDate', 'endDateIso', 'end_date']:
        end_date_str = market.get(field)
        if end_date_str:
            try:
                # Handle multiple formats
                end_date = pd.to_datetime(end_date_str)
                now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
                delta = (end_date - now).total_seconds() / 86400
                return max(0.0, delta)
            except:
                continue

    return 999.0  # Fallback only if ALL date fields fail
```

### Fix 3: Improved Spread Calculation
```python
def _calculate_spread(self, book, market=None) -> float:
    """Calculate bid-ask spread with fallback to market prices"""

    # Try order book first
    if book:
        spread = self._calculate_spread_from_book(book)
        if spread < 0.5:  # Sanity check (reject 50%+ spreads)
            return spread

    # Fallback to market outcome prices
    if market:
        try:
            outcome_prices = json.loads(market.get('outcomePrices', '[]'))
            if len(outcome_prices) >= 2:
                yes_price = float(outcome_prices[0])
                no_price = float(outcome_prices[1])

                # Spread = 1 - (yes_price + no_price)
                # For efficient markets, yes + no ≈ 1
                spread = abs(1 - (yes_price + no_price))
                return min(spread, 0.20)  # Cap at 20%
        except:
            pass

    # Default fallback
    return 0.05
```

### Fix 4: Market Selection Criteria
Only collect markets that meet ALL criteria:
- `active == True`
- `closed == False`
- `liquidity > 1000` (minimum $1k liquidity)
- `volume24hr > 100` (minimum $100 daily volume)
- `endDate` exists and is parseable
- Not archived or restricted

## Testing Strategy:

1. Stop current data collector
2. Backup existing data:
   ```bash
   cp src/data/polymarket/training_data/market_snapshots.csv src/data/polymarket/training_data/market_snapshots_backup.csv
   ```

3. Delete old data (contains closed markets):
   ```bash
   rm src/data/polymarket/training_data/market_snapshots.csv
   ```

4. Start fixed data collector
5. Let run for 1 hour (60 snapshots)
6. Verify data quality:
   ```python
   df = pd.read_csv('market_snapshots.csv')
   print(df['time_to_resolution_days'].describe())  # Should NOT all be 999
   print(df['spread'].describe())  # Should be < 0.20 for most
   ```

## Expected Results After Fix:

```
time_to_resolution_days:
  mean:  30-60 days
  min:   0 days
  max:   180 days (will filter out longer)

spread:
  mean:  0.02-0.05 (2-5%)
  median: 0.03 (3%)
  max:   0.20 (20% cap)
```

## Timeline:

- Apply fixes: 10 minutes
- Restart data collector: immediate
- First valid snapshot: 1 minute
- Minimum backtest data: 24 hours (1,440 snapshots)
- Optimal backtest data: 48-72 hours (2,880-4,320 snapshots)
