# 🤖 AI AGENTS FOR TRADING

<p align="center">
  <a href="https://www.moondev.com/"><img src="moondev.png" width="300" alt="Moon Dev"></a>
</p>

## 🎯 Vision
ai agents are clearly the future and the entire workforce will be replaced or atleast using ai agents. while i am a quant and building agents for algo trading i will be contributing to all different types of ai agent flows and placing all of the agents here for free, 100% open sourced because i beleive code is the great equalizer and we have never seen a regime shift like this so i need to get this code to the people

feel free to join [our discord](https://discord.gg/8UPuVZ53bh) if you beleive ai agents will be integrated into the workforce

⭐️ [first full concise documentation video (watch here)](https://youtu.be/RlqzkSgDKDc)

## Video Updates & Training
📀 follow all updates here on youtube: https://www.youtube.com/playlist?list=PLXrNVMjRZUJg4M4uz52iGd1LhXXGVbIFz

## Live Agents
- **Trading Agent** (`trading_agent.py`): **DUAL-MODE AI trading system** - Toggle between single model (fast ~10s) or swarm mode (6-model consensus ~45-60s). Swarm mode queries Claude 4.5, GPT-5, Gemini 2.5, Grok-4, DeepSeek, and DeepSeek-R1 local for majority vote trading decisions. Configure via `USE_SWARM_MODE` in config.py 
- Strategy Agent (`strategy_agent.py`): Manages and executes trading strategies placed in the strategies folder
- Risk Agent (`risk_agent.py`): Monitors and manages portfolio risk, enforcing position limits and PnL thresholds
- Copy Agent (`copy_agent.py`): monitors copy bot for potential trades
- Whale Agent (`whale_agent.py`): monitors whale activity and announces when a whale enters the market
- Sentiment Agent (`sentiment_agent.py`): analyzes Twitter sentiment for crypto tokens with voice announcements
- Listing Arbitrage Agent (`listingarb_agent.py`): identifies promising Solana tokens on CoinGecko before they reach major exchanges like Binance and Coinbase, using parallel AI analysis for technical and fundamental insights
- Focus Agent (`focus_agent.py`): randomly samples audio during coding sessions to maintain productivity, providing focus scores and voice alerts when focus drops (~$10/month, perfect for voice-to-code workflows)
- Funding Agent (`funding_agent.py`): monitors funding rates across exchanges and uses AI to analyze opportunities, providing voice alerts for extreme funding situations with technical context 🌙
- Liquidation Agent (`liquidation_agent.py`): tracks liquidation events with configurable time windows (15min/1hr/4hr), providing AI analysis and voice alerts for significant liquidation spikes 💦
- Chart Agent (`chartanalysis_agent.py`): looks at any crypto chart and then analyzes it with ai to make a buy/sell/nothing reccomendation.
- funding rate arbitrage agent (`fundingarb_agent.py`): tracks the funding rate on hyper liquid to find funding rate arbitrage opportunities between hl and solana
- rbi agent (`rbi_agent.py`): uses deepseek to research trading strategies based on the youtube video, pdf, or words you give it. then sends to his ai friend who codes out the backtest.
- **Pinescript Conversion System**: Precise TradingView Pinescript → Python backtesting.py conversions with zero loss in translation. Handles complex indicators, custom libraries (loxx), multi-stop systems, and R-based position sizing. Includes data extraction tools for daily NAV and trades list. See `src/data/pinescript_conversions/`
- twitter agent (`tweet_agent.py`): takes in text and creates tweets using deepseek or other models
- video agent (`video_agent.py`): takes in text to create videos by creating audio snippets using elevenlabs and combining with raw_video footage
- new or top tokens (`new_or_top_agent.py`): an agent that looks at the new tokens and the top tokens from coin gecko api
- chat agent (`chat_agent.py`): an agent that monitors youtube live stream chat, moderates & responds to known questions. absolute fire.
- clips agent (`clips_agent.py`): an agent that helps clip long videos into shorter ones so you can upload to your youtube and get paid more info is in the code notes and here: https://discord.gg/XAw8US9aHT
- phone agent (`phone_agent.py`): an ai agent that can take phone calls for you
- sniper agent (`sniper_agent.py`): sniper agent that watches for new solana token launches and will then analyze them and maybe snipe
- tx agent (`tx_agent.py`): watches transactions made by my copy list and then prints them out with an optional auto tab open
- solana agent (`solana_agent.py`): looks at the sniper agent and the tx agent in order to select which memes may be interesting
- million agent (`million_agent.py`): uses million context window from gemini to pull in a knowledge base
- tiktok agent (`tiktok_agent.py`): scrolls tiktok and gets screenshots of the video + comments to extract consumer data in order to feed into algos. sometimes called social arbitrage
- compliance agent (`compliance_agent.py`): analyzes TikTok ads for Facebook advertising compliance, extracting frames and transcribing audio to check against FB guidelines
- research agent (`research_agent`): an agent to fill the ideas.txt so the rbi agent can run forever
- real time clips agent (`src/agents/realtime_clips_agent.py`): an ai agent that makes real time clips of streamers using obs
- housecoin agent (`src/agents/housecoin_agent.py`): DCA (dollar cost average) agent with AI confirmation layer using Grok-4 for the thesis: 1 House = 1 Housecoin 🏠
- swarm agent (`src/agents/swarm_agent.py`): queries 6 AI models in parallel (Claude 4.5, GPT-5, Gemini 2.5, Grok-4, DeepSeek, DeepSeek-R1 local), generates AI consensus summary, returns clean JSON with model mapping for easy parsing 🐝

**⚠️ IMPORTANT: This is an experimental project. There are NO guarantees of profitability. Trading involves substantial risk of loss.**

## ⚠️ Critical Disclaimers

*There is no token associated with this project and there never will be. any token launched is not affiliated with this project, moon dev will never dm you. be careful. don't send funds anywhere*

**PLEASE READ CAREFULLY:**

1. This is an experimental research project, NOT a trading system
2. There are NO plug-and-play solutions for guaranteed profits
3. We do NOT provide trading strategies
4. Success depends entirely on YOUR:
   - Trading strategy
   - Risk management
   - Market research
   - Testing and validation
   - Overall trading approach

5. NO AI agent can guarantee profitable trading
6. You MUST develop and validate your own trading approach
7. Trading involves substantial risk of loss
8. Past performance does not indicate future results

## 👂 Looking for Updates?
Project updates will be posted in Discord, join here: [discord.gg/8UPuVZ53bh](https://discord.gg/8UPuVZ53bh) 

## 🔗 Links
- Free Algo Trading Roadmap: [moondev.com](https://moondev.com)
- Algo Trading Education: [algotradecamp.com](https://algotradecamp.com)
- Business Contact [moon@algotradecamp.com](mailto:moon@algotradecamp.com)

## 🚀 Quick Start Guide

python 3.10.9 is what was used during dev

1. ⭐ **Star the Repo**
   - Click the star button to save it to your GitHub favorites

2. 🍴 **Fork the Repo**
   - Fork to your GitHub account to get your own copy
   - This lets you make changes and track updates

3. 💻 **Open in Your IDE**
   - Clone to your local machine
   - Recommended: Use [Cursor](https://www.cursor.com/) or [Windsurfer](https://codeium.com/) for AI-enabled coding

4. 🔑 **Set Environment Variables**
   - Check `.env.example` for required variables
   - Create a copy of above and name it `.env` file with your keys:
     - Anthropic API key
     - Other trading API keys
   - ⚠️ Never commit or share your API keys!

5. 🤖 **Customize Agent Prompts**
   - Navigate to `/agents` folder
   - Modify LLM prompts to fit your needs
   - Each agent has configurable parameters

6. 📈 **Implement Your Strategies**
   - Add your strategies to `/strategies` folder
   - Remember: Out-of-box code is NOT profitable
   - Thorough testing required before live trading

7. 🏃‍♂️ **Run the System**
   - Execute via `main.py`
   - Toggle agents on/off as needed
   - Monitor logs for performance

---

## 🌊 Swarm Trading System

The **Trading Agent** (`src/agents/trading_agent.py`) is a **fully self-contained, dual-mode AI trading system**.

### ⚙️ Configuration - All in One Place
**ALL settings are at the top of `src/agents/trading_agent.py` (lines 55-95)**

No need to edit multiple files - everything is configured in the trading agent itself:

```python
# 🌊 AI MODE (line 71)
USE_SWARM_MODE = True  # True = swarm, False = single model

# 📈 TRADING MODE (line 75)
LONG_ONLY = True  # True = Long positions only (Solana on-chain, no shorting)
                  # False = Long & Short (HyperLiquid perpetuals)

# 💰 POSITION SIZING (lines 93-96)
usd_size = 25                    # Target position size
max_usd_order_size = 3           # Order chunk size
MAX_POSITION_PERCENTAGE = 30     # Max % per position
CASH_PERCENTAGE = 20             # Min cash buffer

# 📊 MARKET DATA (lines 75-79)
DAYSBACK_4_DATA = 3              # Days of history
DATA_TIMEFRAME = '1H'            # Bar timeframe
# Options: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M

# 🎯 TOKENS (lines 93-96)
# ⚠️ ALL tokens in this list will be analyzed (one at a time)
MONITORED_TOKENS = [
    # '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',  # FART (disabled)
    'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump',     # housecoin (active)
]
```

**⚠️ Important:** The swarm analyzes **ALL tokens** in `MONITORED_TOKENS` one at a time. Each token takes ~60 seconds in swarm mode. Comment out tokens you don't want to trade by adding `#` at the start of the line.

### 🤖 Single Model Mode (Fast)
- Uses one AI model for quick trading decisions
- Response time: ~10 seconds per token
- Best for: Fast execution, high-frequency strategies
- Set `USE_SWARM_MODE = False` and configure `AI_MODEL_TYPE`

### 🌊 Swarm Mode (Consensus)
- Queries **6 AI models simultaneously** for consensus voting
- Response time: ~45-60 seconds per token
- Each model votes: **"Buy"**, **"Sell"**, or **"Do Nothing"**
- Majority decision wins with confidence percentage
- Best for: Higher confidence trades, 15-minute+ timeframes

**Trading Actions:**
- **"Buy"** = Bullish signal (open/maintain long position)
- **"Sell"** = Bearish signal (close long position)
- **"Do Nothing"** = Hold current position unchanged (no action taken)

**Trading Modes:**

Set `LONG_ONLY` to match your trading platform:

- **True (Solana On-Chain - Default):**
  - Long positions only (no shorting available)
  - **"Buy"** = Opens/maintains long position
  - **"Sell"** = Closes long position (exits to cash)
  - **"Sell" with no position** = Does nothing (can't short)
  - **Use case:** Spot trading on Solana, token trading

- **False (HyperLiquid Perpetuals):**
  - Long AND short positions available
  - **"Buy"** = Opens/maintains long position
  - **"Sell"** = Closes long OR opens short position
  - **"Sell" with no position** = Can open short
  - **Use case:** Perpetual futures on HyperLiquid

**Portfolio Allocation:**
- Automatically allocates capital when swarm recommends BUY signals
- Skipped when all signals are SELL or DO NOTHING
- Manages position sizing based on confidence levels

**Active Swarm Models:**
1. **Claude Sonnet 4.5** - Anthropic's latest reasoning model
2. **GPT-5** - OpenAI's most advanced model
3. **Gemini 2.5 Flash** - Google's fast multimodal model
4. **Grok-4 Fast Reasoning** - xAI's 2M context window model
5. **DeepSeek Chat** - DeepSeek API reasoning model
6. **DeepSeek-R1 Local** - Local reasoning model (free!)

**Example Swarm Output:**
```
🌊 Swarm Consensus: BUY with 83% agreement

Swarm Consensus (6 models voted):
  Buy: 5 votes
  Sell: 0 votes
  Do Nothing: 1 vote

Individual votes:
  - claude: Buy
  - openai: Buy
  - gemini: Buy
  - xai: Buy
  - deepseek: Buy
  - ollama: Do Nothing
```

**Market Data Details:**
- Current settings: 3 days @ 1H = **~72 bars per token**
- For 15-min bars: Change `DATA_TIMEFRAME = '15m'` = **~288 bars**
- Each query includes: Full OHLCV dataframe, strategy signals, token metadata

**How to Run:**
```bash
# Edit settings at top of file
vim src/agents/trading_agent.py  # Lines 55-95

# Run standalone
python src/agents/trading_agent.py

# Or via main orchestrator
python src/main.py  # Enable in ACTIVE_AGENTS
```

---
## 🗺️ ROADMAP

### In Progress
- [x] **HyperLiquid Perps Integration** ✅
   - Built ExchangeManager for seamless switching between Solana and HyperLiquid
   - No router needed - using explicit imports via `nice_funcs_hyperliquid.py`

### Coming Soon
- [ ] **Polymarket Integration**
   - Prediction market trading capabilities
- [ ] **Base Chain Integration**
   - Support for Base L2 network trading
- [ ] **Extended Integration**
   - Additional exchange support using same ExchangeManager pattern
- [ ] **HyperLiquid Spot Trading**
   - Spot market support on HyperLiquid
- [ ] **Trending Agent**
   - Spots leaders on HyperLiquid and trades with data + LLM analysis
- [ ] **RBI Agent Updates**
   - Continued improvements to research-based inference agent
- [ ] **postion sizing agent**
   - looks at volume and liquidations to determine position sizing
- [ ] **regime agents**
   - constantly determining the trading regime we are in and running strategies for that regime
- [x] **execution agents** ✅
   - when a signal is triggered ask a swarm of agents if we should abide (IMPLEMENTED: Swarm Mode in Trading Agent)
- [ ] **polymarket sweeper agent**
   - watches our polymarket sweeper dashboard and follows some sweepers

### Future Ideas
- [ ] **Lighter Integration**
- [ ] **Pacifica Integration**
- [ ] **Hibachi Integration**
- [ ] **Aster Integration**
- [ ] **HyperEVM Support**


---
*Built with love by Moon Dev - Pioneering the future of AI-powered trading*


## 📜 Detailed Disclaimer
The content presented is for educational and informational purposes only and does not constitute financial advice. All trading involves risk and may not be suitable for all investors. You should carefully consider your investment objectives, level of experience, and risk appetite before investing.

Past performance is not indicative of future results. There is no guarantee that any trading strategy or algorithm discussed will result in profits or will not incur losses.

**CFTC Disclaimer:** Commodity Futures Trading Commission (CFTC) regulations require disclosure of the risks associated with trading commodities and derivatives. There is a substantial risk of loss in trading and investing.

I am not a licensed financial advisor or a registered broker-dealer. Content & code is based on personal research perspectives and should not be relied upon as a guarantee of success in trading.