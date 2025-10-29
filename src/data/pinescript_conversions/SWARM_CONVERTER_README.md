# Pinescript Swarm Converter - Goal-Oriented Strategy Translation

**Three-Phase AI Swarm System: UNDERSTAND → REFACTOR → VALIDATE**

## Quick Start

### 1. Place Files

```bash
# Your strategy
cp my_strategy.pine src/data/pinescript_conversions/input/strategies/

# Custom indicators (D-Bands, custom MAs, etc.)
cp D_Bands.pine src/data/pinescript_conversions/input/indicators/
cp CustomMA.pine src/data/pinescript_conversions/input/indicators/
```

### 2. Run Converter

```bash
python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine
```

### 3. Review & Approve

System will pause after Phase 1 for your review. Approve to continue.

### 4. Check Results

```bash
# Generated Python
cat src/data/pinescript_conversions/output/strategies/my_strategy.py

# Validation report
cat src/data/pinescript_conversions/validation/my_strategy_validation.json
```

## The Three-Phase Workflow

```
┌──────────────────────────────────────────────────────┐
│  PHASE 1: UNDERSTAND                                 │
│  ────────────────────                                │
│  What is this strategy trying to achieve?            │
│                                                      │
│  • Scan for custom indicator dependencies            │
│  • Load D-Bands.pine, CustomMA.pine, etc.           │
│  • Claude analyzes GOALS and INTENT                  │
│  • Generate understanding report                     │
│  • → USER REVIEW GATE ←                             │
└──────────────────────────────────────────────────────┘
                        ↓
                   [Approve?]
                        ↓
┌──────────────────────────────────────────────────────┐
│  PHASE 2: REFACTOR                                   │
│  ──────────────────                                  │
│  Convert with goal preservation                      │
│                                                      │
│  • DeepSeek generates Python code                    │
│  • Implements D-Bands EXACTLY as specified           │
│  • Preserves strategy goals                          │
│  • Matches parameter defaults                        │
│  • Generates backtesting.py code                     │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  PHASE 3: VALIDATE                                   │
│  ───────────────────                                 │
│  Multi-model consensus check                         │
│                                                      │
│  • Claude: Does Python achieve same goals?           │
│  • GPT-4: Are parameters correct?                    │
│  • DeepSeek: Are indicators right?                   │
│  • Consensus: 2/3 must PASS                          │
└──────────────────────────────────────────────────────┘
```

## Why This Approach?

### Problem: Naive Conversion

Your strategy uses `D_Bands()` - a proprietary indicator you built.

**Bad converter assumes it's Bollinger Bands:**
```python
# WRONG! Lost your edge!
upper, middle, lower = ta.BBANDS(close, 20, 2.0)
```

### Solution: Goal-Oriented + Indicator Library

**Phase 1**: System loads `D_Bands.pine`, understands how it works

**Phase 2**: Implements D-Bands EXACTLY from your specification

**Phase 3**: 3 AI models validate: "Is D-Bands correct?" → YES

**Result**: Python achieves SAME GOALS as Pinescript

## Folder Structure

```
src/data/pinescript_conversions/
├── input/
│   ├── strategies/          ← Place .pine strategies here
│   │   └── my_strategy.pine
│   └── indicators/          ← Place custom indicators here
│       ├── D_Bands.pine
│       └── CustomMA.pine
├── output/
│   ├── strategies/          → Generated Python
│   │   └── my_strategy.py
│   └── indicators/          → (Future: indicator library)
├── analysis/                → Phase 1 reports
│   └── my_strategy_analysis.json
└── validation/              → Phase 3 reports
    └── my_strategy_validation.json
```

## Example Session

```
$ python src/agents/pinescript_swarm_converter.py --strategy adx_d_bands.pine

================================================================================
PINESCRIPT SWARM CONVERTER
================================================================================

[LOAD] Loading strategy: adx_d_bands.pine
[SCAN] Scanning for custom indicator dependencies...
[SCAN] Found 2 dependencies:
       - D_Bands
       - ADX_Custom
[LOAD] Loading indicator: D_Bands...
[LOAD] ✓ Loaded: D_Bands.pine
[LOAD] Loading indicator: ADX_Custom...
[LOAD] ✓ Loaded: ADX_Custom.pine

================================================================================
PHASE 1: UNDERSTAND
================================================================================

[PHASE 1] Analyzing strategy with Claude...
[PHASE 1] ✓ Understanding complete

[SAVE] Analysis: src/data/pinescript_conversions/analysis/adx_d_bands_analysis.json

================================================================================
PHASE 1 COMPLETE - REVIEW REQUIRED
================================================================================

Strategy Understanding:
{
  "core_thesis": "Capture momentum breakouts with volatility confirmation",
  "entry_conditions": [
    "ADX > 20 (strong trend)",
    "Price breaks above upper D-Band",
    "+DI > -DI (uptrend direction)"
  ],
  "exit_conditions": [
    "Price closes below middle D-Band",
    "ADX drops below 15"
  ],
  "indicator_roles": {
    "D_Bands": "Proprietary double-WMA bands with asymmetric multipliers",
    "ADX_Custom": "Modified ADX with custom smoothing"
  },
  "parameters": {
    "d_bands_length": {"default": 20, "purpose": "Band calculation period"},
    "d_bands_mult": {"default": 2.5, "purpose": "Upper band multiplier"},
    "adx_threshold": {"default": 20, "purpose": "Minimum trend strength"}
  }
}

Proceed with Phase 2 (Refactor)? [y/N]: y

================================================================================
PHASE 2: REFACTOR
================================================================================

[PHASE 2] Converting to Python with DeepSeek...
[PHASE 2] ✓ Refactor complete

[SAVE] Python code: src/data/pinescript_conversions/output/strategies/adx_d_bands.py

================================================================================
PHASE 3: VALIDATE
================================================================================

[VALIDATE] claude reviewing...
[VALIDATE] ✓ claude complete

[VALIDATE] gpt4 reviewing...
[VALIDATE] ✓ gpt4 complete

[VALIDATE] deepseek reviewing...
[VALIDATE] ✓ deepseek complete

[VALIDATE] Consensus: PASS (3/3 models passed)

[SAVE] Validation: src/data/pinescript_conversions/validation/adx_d_bands_validation.json

================================================================================
CONVERSION COMPLETE
================================================================================

Input:      src/data/pinescript_conversions/input/strategies/adx_d_bands.pine
Output:     src/data/pinescript_conversions/output/strategies/adx_d_bands.py
Analysis:   src/data/pinescript_conversions/analysis/adx_d_bands_analysis.json
Validation: src/data/pinescript_conversions/validation/adx_d_bands_validation.json

Consensus:  PASS (3/3)

[SUCCESS] Conversion validated by swarm
```

## Command Options

**Basic conversion** (with review gate):
```bash
python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine
```

**Auto-approve** (skip review):
```bash
python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine --auto-approve
```

**Help:**
```bash
python src/agents/pinescript_swarm_converter.py --help
```

## Understanding Phase 1 Analysis

The analysis JSON contains:

```json
{
  "core_thesis": "Brief description of strategy goal",
  "entry_conditions": ["Condition 1", "Condition 2", ...],
  "exit_conditions": ["Condition 1", "Condition 2", ...],
  "position_sizing": "Method description",
  "risk_management": ["Rule 1", "Rule 2", ...],
  "indicator_roles": {
    "D_Bands": "How it's used and why",
    "CustomMA": "How it contributes"
  },
  "parameters": {
    "param1": {"default": 20, "purpose": "Why it matters"},
    "param2": {"default": 2.5, "purpose": "What it controls"}
  }
}
```

**Review this carefully!** If the understanding is wrong, reject and fix indicator files.

## Understanding Phase 3 Validation

Each model independently answers:

```json
{
  "model": "claude",
  "validation": {
    "same_goals": "Yes/No + explanation",
    "entry_correct": "Yes/No + details",
    "exit_correct": "Yes/No + details",
    "parameters_match": "Yes/No + list mismatches",
    "indicators_correct": "Yes/No + concerns",
    "lost_in_translation": "What was lost (if anything)",
    "overall_grade": "PASS / PARTIAL / FAIL"
  }
}
```

**Consensus**: 2 out of 3 models must give "PASS"

## Custom Indicators

The system correctly handles:

- **D-Bands** (your proprietary indicator)
- **Custom moving averages** (ZLEMA, HMA, etc.)
- **Loxx library indicators**
- **Proprietary oscillators**
- **Complex multi-component indicators**

**Critical requirement**: Place `.pine` source in `input/indicators/` folder!

### File Naming (Flexible)

All these work:
- `D_Bands.pine`
- `d_bands.pine`
- `d-bands.pine`
- `DBands.pine`

System matches:
- Exact name
- Case-insensitive
- Underscore/dash variations

## Troubleshooting

### "Indicator not found"

```
[WARN] ✗ Not found: D_Bands.pine
[WARN]   Please add to: src/data/pinescript_conversions/input/indicators/
```

**Solution**: Copy `D_Bands.pine` to the indicators folder

### "Validation failed: 1/3 pass"

```
[VALIDATE] Consensus: FAIL (1/3 models passed)
```

**Check validation report:**
```bash
cat src/data/pinescript_conversions/validation/my_strategy_validation.json
```

**Common causes:**
- Missing indicator source (incomplete conversion)
- Pinescript feature not supported by backtesting.py
- Ambiguous strategy logic

### "Stopped after Phase 1"

You rejected during review. This is good! Fix the issue before continuing:

1. Check if indicators are loaded
2. Review understanding JSON
3. Fix indicator files if needed
4. Re-run conversion

## Comparison: Simple vs Swarm Converter

This repo has TWO converters:

| Feature | pinescript_converter_agent.py | pinescript_swarm_converter.py |
|---------|------------------------------|-------------------------------|
| **Focus** | R-based position sizing | Goal-oriented conversion |
| **Understanding phase** | ✗ No | ✓ Yes (Phase 1) |
| **Indicator library** | ✗ No | ✓ Yes |
| **Validation** | ✗ No | ✓ Yes (3-model swarm) |
| **User review** | ✗ No | ✓ Yes (after Phase 1) |
| **Best for** | Simple strategies | Complex + custom indicators |

**Use Swarm Converter when:**
- Strategy uses custom indicators
- You want high confidence in correctness
- Strategy is complex with multiple components
- You have proprietary logic (D-Bands, etc.)

## Cost & Time

**Cost per conversion**:
- Phase 1 (Claude): ~$0.02
- Phase 2 (DeepSeek): ~$0.01
- Phase 3 (3 models): ~$0.06
- **Total**: ~$0.09 per strategy

**Time**: 2-5 minutes depending on complexity

## Next Steps After Conversion

1. **Review validation report**
2. **Run backtest**: `python output/strategies/my_strategy.py`
3. **Compare with Pinescript** (if you have TV results)
4. **Extract trades**: `python extract_backtest_data.py`
5. **Iterate if needed**: Fix and re-run

## Integration with Existing System

Generated strategies are compatible with:

- **Backtesting**: Run directly
- **RBI Agent**: Same patterns
- **Strategy Agent**: Adapt for live trading
- **Data files**: Uses `src/data/rbi/BTC-USD-*.csv`

## Technical Details

**Models Used**:
- **Understanding**: Claude Sonnet 4.5 (deep reasoning)
- **Refactor**: DeepSeek (fast code generation)
- **Validate**: Claude + GPT-4 + DeepSeek (consensus)

**Token Limits**:
- Phase 1: 4,000 tokens
- Phase 2: 6,000 tokens
- Phase 3: 2,000 tokens per model

**Exit Codes**:
- `0` = Success (PASS consensus)
- `1` = Warning (conversion complete but concerns exist)
- `2` = Stopped after Phase 1 review
- `3` = Failed (error during conversion)

## Support

For issues:
1. Check validation report first
2. Review Phase 1 analysis
3. Verify indicator files are present
4. Ask in Discord/GitHub with analysis + validation JSONs attached

---

**Built by Moon Dev** 🚀

*"Understand the goal, then convert with confidence"*
