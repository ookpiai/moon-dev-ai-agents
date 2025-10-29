"""
Pinescript Swarm Converter - Goal-Oriented Strategy Translation

Three-Phase Workflow:
1. UNDERSTAND: Deep analysis of strategy goals + all custom indicators
2. REFACTOR: Convert to Python preserving goals (not just syntax)
3. VALIDATE: Multi-model swarm consensus

Folder Structure:
- input/strategies/    → Place .pine strategy files here
- input/indicators/    → Place custom indicators (D-Bands, etc.) here
- output/strategies/   → Generated Python strategies
- output/indicators/   → Generated Python indicators
- analysis/            → Phase 1 understanding reports
- validation/          → Phase 3 validation reports

Usage:
    python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine
"""

import sys
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from termcolor import cprint
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import ModelFactory


class PinescriptSwarmConverter:
    """
    Goal-oriented Pinescript to Python converter with swarm validation.

    Key Principles:
    - Understand GOALS before converting
    - Resolve ALL indicator dependencies
    - Multi-model consensus validation
    - Human review gates
    """

    def __init__(self):
        self.input_strategies_dir = Path("src/data/pinescript_conversions/input/strategies")
        self.input_indicators_dir = Path("src/data/pinescript_conversions/input/indicators")
        self.output_strategies_dir = Path("src/data/pinescript_conversions/output/strategies")
        self.output_indicators_dir = Path("src/data/pinescript_conversions/output/indicators")
        self.analysis_dir = Path("src/data/pinescript_conversions/analysis")
        self.validation_dir = Path("src/data/pinescript_conversions/validation")

        # Ensure directories exist
        for directory in [self.input_strategies_dir, self.input_indicators_dir,
                          self.output_strategies_dir, self.output_indicators_dir,
                          self.analysis_dir, self.validation_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize LLM models
        factory = ModelFactory()
        self.understanding_model = factory.get_model('claude')  # Claude for deep analysis
        self.refactor_model = factory.get_model('claude')  # Claude for code generation (DeepSeek connection issues)
        self.validation_models = [
            ('claude', factory.get_model('claude')),
            ('gemini', factory.get_model('gemini')),  # Gemini instead of o1-mini (API compatibility)
            ('claude', factory.get_model('claude'))   # Extra Claude for consensus
        ]

        cprint("\n[INIT] Pinescript Swarm Converter initialized", "green")
        cprint(f"[INIT] Input strategies: {self.input_strategies_dir}", "cyan")
        cprint(f"[INIT] Input indicators: {self.input_indicators_dir}", "cyan")

    def scan_dependencies(self, strategy_code: str) -> List[str]:
        """
        Scan Pinescript strategy for custom indicator dependencies.

        Detects patterns like:
        - library("indicator_name")
        - import indicator_name
        - CustomFunction() calls
        """
        cprint("\n[SCAN] Scanning for custom indicator dependencies...", "cyan")

        dependencies = set()

        # Pattern 1: library imports
        lib_pattern = r'library\("([^"]+)"\)'
        lib_matches = re.findall(lib_pattern, strategy_code)
        dependencies.update(lib_matches)

        # Pattern 2: import statements
        import_pattern = r'import\s+([^\s,]+)'
        import_matches = re.findall(import_pattern, strategy_code)
        dependencies.update(import_matches)

        # Pattern 3: Custom function calls (CamelCase pattern)
        # Look for calls like: D_Bands(), CustomMA(), MyIndicator()
        func_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\s*\('
        func_matches = re.findall(func_pattern, strategy_code)

        # Filter out built-in Pinescript functions
        builtin_functions = {
            'Open', 'High', 'Low', 'Close', 'Volume', 'Time',
            'Array', 'Matrix', 'Table', 'Line', 'Label', 'Box',
            'Color', 'String', 'Int', 'Float', 'Bool'
        }

        custom_funcs = [f for f in func_matches if f not in builtin_functions]
        dependencies.update(custom_funcs)

        dependencies_list = sorted(list(dependencies))

        if dependencies_list:
            cprint(f"[SCAN] Found {len(dependencies_list)} dependencies:", "green")
            for dep in dependencies_list:
                cprint(f"       - {dep}", "yellow")
        else:
            cprint("[SCAN] No custom dependencies found (strategy uses only built-ins)", "yellow")

        return dependencies_list

    def load_indicator_code(self, indicator_name: str) -> Optional[str]:
        """
        Load indicator Pinescript code from input/indicators/ folder.

        Tries:
        1. Exact match: {indicator_name}.pine
        2. Case-insensitive match
        3. Underscore variations: D_Bands → d-bands, d_bands, DBands
        """
        cprint(f"[LOAD] Loading indicator: {indicator_name}...", "cyan")

        # Try exact match
        indicator_file = self.input_indicators_dir / f"{indicator_name}.pine"

        if not indicator_file.exists():
            # Try case-insensitive
            for file in self.input_indicators_dir.glob("*.pine"):
                if file.stem.lower() == indicator_name.lower():
                    indicator_file = file
                    break

        if not indicator_file.exists():
            # Try underscore/dash variations
            normalized = indicator_name.replace('_', '').replace('-', '').lower()
            for file in self.input_indicators_dir.glob("*.pine"):
                file_normalized = file.stem.replace('_', '').replace('-', '').lower()
                if file_normalized == normalized:
                    indicator_file = file
                    break

        if indicator_file.exists():
            cprint(f"[LOAD] OK Loaded: {indicator_file.name}", "green")
            return indicator_file.read_text(encoding='utf-8')
        else:
            cprint(f"[WARN] X Not found: {indicator_name}.pine", "red")
            cprint(f"[WARN]   Please add to: {self.input_indicators_dir}", "yellow")
            return None

    def phase1_understand(self, strategy_code: str, dependencies: List[str]) -> Dict:
        """
        Phase 1: UNDERSTAND - Deep analysis of strategy goals.

        Questions answered:
        - What is the core trading thesis?
        - What conditions trigger entries?
        - What conditions close positions?
        - How are custom indicators used?
        - What are the key parameters?
        """
        cprint("\n" + "="*80, "cyan", attrs=['bold'])
        cprint("PHASE 1: UNDERSTAND", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        # Load all dependency code
        indicator_context = {}
        missing_indicators = []

        for dep in dependencies:
            code = self.load_indicator_code(dep)
            if code:
                indicator_context[dep] = code
            else:
                missing_indicators.append(dep)

        if missing_indicators:
            cprint(f"\n[WARN] Missing {len(missing_indicators)} indicator(s)!", "red", attrs=['bold'])
            cprint("[WARN] Please add these files to input/indicators/:", "yellow")
            for ind in missing_indicators:
                cprint(f"       - {ind}.pine", "red")
            cprint("\n[WARN] Proceeding with partial understanding...", "yellow")

        # Build comprehensive understanding prompt
        system_prompt = """You are an expert trading strategy analyst.

Your task: Deeply understand what a Pinescript strategy is trying to achieve.

Focus on GOALS and INTENT, not just syntax.

Analyze:
1. CORE THESIS: What market behavior is being exploited?
2. ENTRY LOGIC: What conditions trigger trades?
3. EXIT LOGIC: What conditions close positions?
4. POSITION SIZING: How are sizes determined?
5. RISK MANAGEMENT: Stops, targets, trailing logic?
6. CUSTOM INDICATORS: How do they contribute to the thesis?
7. KEY PARAMETERS: Defaults and their significance?

Provide structured JSON response:
{
  "core_thesis": "Brief description of what strategy is trying to achieve",
  "entry_conditions": ["Condition 1", "Condition 2", ...],
  "exit_conditions": ["Condition 1", "Condition 2", ...],
  "position_sizing": "Method description",
  "risk_management": ["Rule 1", "Rule 2", ...],
  "indicator_roles": {
    "IndicatorName": "How it's used and why"
  },
  "parameters": {
    "param1": {"default": value, "purpose": "why it matters"}
  }
}"""

        user_content = f"""# Strategy Code

```pinescript
{strategy_code}
```

"""

        if indicator_context:
            user_content += "\n# Custom Indicators Referenced\n\n"
            for name, code in indicator_context.items():
                user_content += f"## {name}\n\n```pinescript\n{code}\n```\n\n"

        user_content += """
Analyze this strategy and provide structured understanding in JSON format as specified."""

        # Get deep understanding
        cprint("[PHASE 1] Analyzing strategy with Claude...", "cyan")

        try:
            response_obj = self.understanding_model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.1,
                max_tokens=4000
            )

            # Extract text from ModelResponse object
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = {"raw_analysis": response}

            cprint("[PHASE 1] OK Understanding complete", "green")

        except Exception as e:
            cprint(f"[PHASE 1] ERROR Analysis failed: {e}", "red")
            analysis = {"error": str(e), "raw_response": response if 'response' in locals() else None}

        return {
            "analysis": analysis,
            "dependencies": dependencies,
            "indicator_code": indicator_context,
            "missing_indicators": missing_indicators,
            "timestamp": datetime.now().isoformat()
        }

    def phase2_refactor(self, strategy_code: str, understanding: Dict) -> str:
        """
        Phase 2: REFACTOR - Convert to Python preserving goals.

        Uses understanding from Phase 1 to ensure Python version achieves
        the SAME GOALS as Pinescript.
        """
        cprint("\n" + "="*80, "cyan", attrs=['bold'])
        cprint("PHASE 2: REFACTOR", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        system_prompt = """You are an expert at converting TradingView Pinescript to Python.

CRITICAL PRINCIPLE: Preserve the GOALS and INTENT, not just syntax.

RULES:
1. USE backtesting.py library
2. USE pandas_ta OR talib for indicators (NOT backtesting.py built-ins)
3. Implement custom indicators EXACTLY as specified (don't substitute!)
4. Match parameter defaults from Pinescript
5. Include comprehensive debug prints
6. Use this template:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import numpy as np

class ConvertedStrategy(Strategy):
    # Parameters (match Pinescript defaults)
    param1 = 10
    param2 = 2.0

    def init(self):
        # Calculate indicators using pandas_ta/talib
        # For custom indicators, implement calculation here
        pass

    def next(self):
        # Entry logic
        if not self.position:
            if <entry_conditions>:
                self.buy()

        # Exit logic
        elif self.position:
            if <exit_conditions>:
                self.position.close()

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("src/data/rbi/BTC-USD-15m.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Run backtest
    bt = Backtest(df, ConvertedStrategy, cash=10000, commission=0.001)
    stats = bt.run()
    print(stats)
    bt.plot()
```

IMPORTANT: If custom indicators like D-Bands are used, implement them from the provided Pinescript - do NOT substitute with wrong indicators!"""

        user_content = f"""# Original Pinescript

```pinescript
{strategy_code}
```

# Strategy Understanding

{json.dumps(understanding['analysis'], indent=2)}

# Custom Indicators

"""

        if understanding['indicator_code']:
            for name, code in understanding['indicator_code'].items():
                user_content += f"## {name}\n\n```pinescript\n{code}\n```\n\n"
        else:
            user_content += "No custom indicators provided.\n\n"

        user_content += """
Convert to Python using backtesting.py. Preserve GOALS, not just syntax.

Output ONLY Python code (no markdown, no explanations)."""

        # Generate Python code
        cprint("[PHASE 2] Converting to Python with DeepSeek...", "cyan")

        try:
            python_code_obj = self.refactor_model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.2,
                max_tokens=6000
            )

            # Extract text from ModelResponse object
            python_code = python_code_obj.content if hasattr(python_code_obj, 'content') else str(python_code_obj)

            # Clean markdown if present
            code_match = re.search(r'```python\n(.*?)```', python_code, re.DOTALL)
            if code_match:
                python_code = code_match.group(1)

            cprint("[PHASE 2] OK Refactor complete", "green")

        except Exception as e:
            cprint(f"[PHASE 2] ERROR Conversion failed: {e}", "red")
            python_code = f"# ERROR: Conversion failed\n# {str(e)}"

        return python_code

    def phase3_validate(self, strategy_code: str, python_code: str, understanding: Dict) -> Dict:
        """
        Phase 3: VALIDATE - Swarm consensus check.

        Each model independently answers:
        1. Does Python achieve same goals as Pinescript?
        2. Are parameters correct?
        3. Are indicators implemented correctly?
        4. What was lost in translation?
        5. Overall grade: PASS / PARTIAL / FAIL
        """
        cprint("\n" + "="*80, "cyan", attrs=['bold'])
        cprint("PHASE 3: VALIDATE", "cyan", attrs=['bold'])
        cprint("="*80, "cyan")

        validation_prompt = f"""Compare Pinescript strategy with Python conversion.

# Original Pinescript
```pinescript
{strategy_code}
```

# Python Conversion
```python
{python_code}
```

# Strategy Understanding
{json.dumps(understanding['analysis'], indent=2)}

VALIDATION QUESTIONS:
1. Does Python achieve SAME GOALS? (Yes/No + explanation)
2. Entry conditions correct? (Yes/No + details)
3. Exit conditions correct? (Yes/No + details)
4. Parameters match defaults? (Yes/No + mismatches)
5. Custom indicators correct? (Yes/No + concerns)
6. What was lost in translation?
7. Overall grade: PASS / PARTIAL / FAIL

Provide structured JSON."""

        validations = []

        for model_name, model in self.validation_models:
            cprint(f"\n[VALIDATE] {model_name} reviewing...", "yellow")

            try:
                response_obj = model.generate_response(
                    system_prompt="You are a critical code reviewer for trading strategy conversions.",
                    user_content=validation_prompt,
                    temperature=0.1,
                    max_tokens=2000
                )

                # Extract text from ModelResponse object
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

                # Extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    validation = json.loads(json_match.group())
                else:
                    validation = {"raw_response": response}

                validations.append({
                    "model": model_name,
                    "validation": validation,
                    "timestamp": datetime.now().isoformat()
                })

                cprint(f"[VALIDATE] OK {model_name} complete", "green")

            except Exception as e:
                cprint(f"[VALIDATE] ERROR {model_name} failed: {e}", "red")
                validations.append({
                    "model": model_name,
                    "error": str(e)
                })

        # Calculate consensus
        grades = []
        for v in validations:
            if 'validation' in v:
                grade = v['validation'].get('overall_grade', 'UNKNOWN')
                grades.append(grade)

        pass_count = grades.count('PASS')
        total_count = len(grades)
        consensus = "PASS" if pass_count >= 2 else "FAIL"

        cprint(f"\n[VALIDATE] Consensus: {consensus} ({pass_count}/{total_count} models passed)",
               "green" if consensus == "PASS" else "red", attrs=['bold'])

        return {
            "consensus": consensus,
            "pass_count": pass_count,
            "total_count": total_count,
            "validations": validations,
            "timestamp": datetime.now().isoformat()
        }

    def convert_strategy(self, strategy_filename: str, auto_approve: bool = False) -> Dict:
        """
        Main conversion workflow: UNDERSTAND → REFACTOR → VALIDATE

        Args:
            strategy_filename: Name of .pine file in input/strategies/
            auto_approve: Skip user review gate (default: False)

        Returns:
            Dict with conversion results and file paths
        """
        cprint("\n" + "="*80, "cyan", attrs=['bold'])
        cprint("PINESCRIPT SWARM CONVERTER", "cyan", attrs=['bold'])
        cprint("="*80 + "\n", "cyan", attrs=['bold'])

        # Load strategy file
        strategy_file = self.input_strategies_dir / strategy_filename
        if not strategy_file.exists():
            cprint(f"[ERROR] Strategy file not found: {strategy_file}", "red")
            return {"error": "File not found", "path": str(strategy_file)}

        cprint(f"[LOAD] Loading strategy: {strategy_filename}", "green")
        strategy_code = strategy_file.read_text(encoding='utf-8')

        # Scan dependencies
        dependencies = self.scan_dependencies(strategy_code)

        # PHASE 1: Understand
        understanding = self.phase1_understand(strategy_code, dependencies)

        # Save analysis
        analysis_file = self.analysis_dir / f"{strategy_file.stem}_analysis.json"
        analysis_file.write_text(json.dumps(understanding, indent=2), encoding='utf-8')
        cprint(f"\n[SAVE] Analysis: {analysis_file}", "green")

        # USER REVIEW GATE
        if not auto_approve:
            cprint("\n" + "="*80, "yellow", attrs=['bold'])
            cprint("PHASE 1 COMPLETE - REVIEW REQUIRED", "yellow", attrs=['bold'])
            cprint("="*80 + "\n", "yellow")

            print("Strategy Understanding:")
            print(json.dumps(understanding['analysis'], indent=2))

            if understanding['missing_indicators']:
                cprint("\n[WARNING] Missing indicators:", "red", attrs=['bold'])
                for ind in understanding['missing_indicators']:
                    cprint(f"  - {ind}.pine", "red")
                cprint("\nConversion will proceed but may be incomplete.", "yellow")

            response = input("\nProceed with Phase 2 (Refactor)? [y/N]: ")
            if response.lower() != 'y':
                cprint("\n[STOP] Conversion stopped by user", "red")
                return {
                    "status": "stopped_after_phase1",
                    "analysis_file": str(analysis_file)
                }

        # PHASE 2: Refactor
        python_code = self.phase2_refactor(strategy_code, understanding)

        # Save Python code
        output_file = self.output_strategies_dir / f"{strategy_file.stem}.py"
        output_file.write_text(python_code, encoding='utf-8')
        cprint(f"\n[SAVE] Python code: {output_file}", "green")

        # PHASE 3: Validate
        validation_results = self.phase3_validate(strategy_code, python_code, understanding)

        # Save validation
        validation_file = self.validation_dir / f"{strategy_file.stem}_validation.json"
        validation_file.write_text(json.dumps(validation_results, indent=2), encoding='utf-8')
        cprint(f"\n[SAVE] Validation: {validation_file}", "green")

        # Final summary
        cprint("\n" + "="*80, "cyan", attrs=['bold'])
        cprint("CONVERSION COMPLETE", "cyan", attrs=['bold'])
        cprint("="*80 + "\n", "cyan", attrs=['bold'])

        cprint(f"Input:      {strategy_file}", "white")
        cprint(f"Output:     {output_file}", "white")
        cprint(f"Analysis:   {analysis_file}", "white")
        cprint(f"Validation: {validation_file}", "white")
        cprint(f"\nConsensus:  {validation_results['consensus']} ({validation_results['pass_count']}/{validation_results['total_count']})",
               "green" if validation_results['consensus'] == "PASS" else "red",
               attrs=['bold'])

        return {
            "status": "complete",
            "consensus": validation_results['consensus'],
            "input_file": str(strategy_file),
            "output_file": str(output_file),
            "analysis_file": str(analysis_file),
            "validation_file": str(validation_file)
        }


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Pinescript strategies to Python with swarm validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with review gate
  python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine

  # Auto-approve (skip review)
  python src/agents/pinescript_swarm_converter.py --strategy my_strategy.pine --auto-approve

Workflow:
  1. Place strategy in: src/data/pinescript_conversions/input/strategies/
  2. Place custom indicators in: src/data/pinescript_conversions/input/indicators/
  3. Run converter
  4. Review analysis (Phase 1)
  5. Approve to continue (Phase 2-3)
  6. Check validation results
        """
    )
    parser.add_argument("--strategy", required=True, help="Strategy filename (e.g., my_strategy.pine)")
    parser.add_argument("--auto-approve", action="store_true", help="Skip user review gate")

    args = parser.parse_args()

    converter = PinescriptSwarmConverter()
    result = converter.convert_strategy(args.strategy, auto_approve=args.auto_approve)

    # Exit codes
    if result.get("status") == "complete":
        if result.get("consensus") == "PASS":
            cprint("\n[SUCCESS] Conversion validated by swarm", "green", attrs=['bold'])
            sys.exit(0)
        else:
            cprint("\n[WARNING] Conversion complete but validation concerns exist", "yellow", attrs=['bold'])
            sys.exit(1)
    elif result.get("status") == "stopped_after_phase1":
        cprint("\n[INFO] Stopped after Phase 1 review", "yellow")
        sys.exit(2)
    else:
        cprint("\n[FAILED] Conversion failed", "red", attrs=['bold'])
        sys.exit(3)


if __name__ == "__main__":
    main()
