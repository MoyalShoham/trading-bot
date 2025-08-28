"""
Advisor module for the crypto trading bot.
Queries OpenAI LLM with recent candles + indicators + position info.
Returns JSON classification for market regime analysis.
"""

import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from config import config
from ai_performance_tracker import ai_performance_tracker
from logger import logger

class MarketAdvisor:
    """AI-powered market regime advisor using OpenAI."""
    
    def __init__(self):
        """Initialize the market advisor."""
        self.api_key = config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _create_prompt(
        self, 
        symbol: str, 
        indicators: Dict[str, any], 
        position_info: Optional[Dict] = None
    ) -> str:
        """
        Create the prompt for the OpenAI API.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators data
            position_info: Current position information
            
        Returns:
            Formatted prompt string
        """
        # Get latest indicator values
        latest_values = {}
        for name, series in indicators.items():
            if hasattr(series, 'iloc') and not series.empty:
                latest_values[name] = float(series.iloc[-1])
        
        # Get AI performance feedback
        performance_context = ai_performance_tracker.get_prompt_enhancement()
        
        prompt = f"""
You are a PROFITABLE cryptocurrency trading expert with 95% accuracy. Your analysis directly controls real money trades.

TARGET: MAXIMIZE PROFITS with calculated risks. Be aggressive when opportunities are clear.

MARKET DATA FOR {symbol}:
{json.dumps(latest_values, indent=2)}

CURRENT POSITION: {json.dumps(position_info or {}, indent=2)}

{performance_context}

PERFORMANCE CONTEXT:
- Current portfolio needs MORE profitable signals
- Previous "uncertain" classifications reduce trading opportunities
- Risk/Reward target: 2:1 minimum
- Win rate target: 60%+

CLASSIFICATION RULES (BE MORE DECISIVE):
- "trend-up": RSI > 55 + EMA alignment + momentum confirmation → STRONG BUY signal
- "trend-down": RSI < 45 + EMA bearish + momentum down → STRONG SELL signal  
- "mean-reversion": RSI 30-70 + tight Bollinger Bands + oscillating price → SCALP opportunity
- "chop": Only use for genuinely conflicting signals (avoid overuse)
- "uncertain": MINIMIZE - only for truly unclear situations

CONFIDENCE SCORING (AIM HIGHER):
- 0.85-1.0: Clear trend + strong momentum + volume confirmation
- 0.70-0.84: Good signals with minor conflicting indicators  
- 0.60-0.69: Moderate signals, acceptable for execution
- Below 0.60: Mark as uncertain only if truly unclear

PROFIT FOCUS: Favor trend-following over uncertainty. Better to take calculated risks than miss opportunities.

RESPONSE FORMAT (JSON only):
{{
    "symbol": "{symbol}",
    "timeframe_exec": "1m",
    "timeframe_confirm": "5m", 
    "regime": "trend-up|trend-down|mean-reversion|chop|uncertain",
    "factors": ["factor1", "factor2", "factor3"],
    "note": "profit opportunity or risk warning",
    "confidence": 0.85
}}

Respond with ONLY the JSON, no additional text.
"""
        return prompt
    
    async def _query_openai(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Query the OpenAI API.
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            OpenAI response or None if failed
        """
        if not self.session:
            logger.log_error("Session not initialized")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cryptocurrency market analyst. Provide analysis in JSON format only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            start_time = time.time()
            
            async with self.session.post(
                self.base_url, 
                headers=headers, 
                json=data
            ) as response:
                
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    logger.log_info(f"OpenAI response received in {response_time:.2f}s")
                    return {
                        'content': content,
                        'response_time': response_time
                    }
                else:
                    error_text = await response.text()
                    logger.log_error(f"OpenAI API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.log_error(f"Error querying OpenAI: {str(e)}")
            return None
    
    def _parse_advisor_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse and validate the advisor response.
        
        Args:
            response: Raw response string from OpenAI
            
        Returns:
            Parsed and validated response or None if invalid
        """
        try:
            # Try to extract JSON from the response
            response = response.strip()
            
            # Remove any markdown formatting if present
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            parsed = json.loads(response)
            
            # Validate required fields
            required_fields = ['symbol', 'regime', 'factors', 'confidence']
            for field in required_fields:
                if field not in parsed:
                    logger.log_error(f"Missing required field: {field}")
                    return None
            
            # Validate regime values
            valid_regimes = ['trend-up', 'trend-down', 'mean-reversion', 'chop', 'uncertain']
            if parsed['regime'] not in valid_regimes:
                logger.log_error(f"Invalid regime: {parsed['regime']}")
                return None
            
            # Validate confidence range
            if not (0.0 <= parsed['confidence'] <= 1.0):
                logger.log_error(f"Invalid confidence value: {parsed['confidence']}")
                return None
            
            # Validate factors
            if not isinstance(parsed['factors'], list) or len(parsed['factors']) < 1:
                logger.log_error("Factors must be a non-empty list")
                return None
            
            logger.log_info(f"Successfully parsed advisor response for {parsed['symbol']}")
            return parsed
            
        except json.JSONDecodeError as e:
            logger.log_error(f"Failed to parse JSON response: {str(e)}")
            return None
        except Exception as e:
            logger.log_error(f"Error parsing advisor response: {str(e)}")
            return None
    
    async def get_market_regime(
        self, 
        symbol: str, 
        indicators: Dict[str, any], 
        position_info: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get market regime classification from OpenAI.
        
        Args:
            symbol: Trading symbol
            indicators: Technical indicators data
            position_info: Current position information
            
        Returns:
            Market regime classification or None if failed
        """
        try:
            # Create the prompt
            prompt = self._create_prompt(symbol, indicators, position_info)
            
            # Query OpenAI
            openai_response = await self._query_openai(prompt)
            if not openai_response:
                return None
            
            # Parse the response
            parsed_response = self._parse_advisor_response(openai_response['content'])
            if not parsed_response:
                return None
            
            # Add metadata
            parsed_response['timestamp'] = datetime.now().isoformat()
            parsed_response['response_time'] = openai_response['response_time']
            
            # Log the advisor response
            logger.log_advisor(parsed_response)
            
            return parsed_response
            
        except Exception as e:
            logger.log_error(f"Error in get_market_regime: {str(e)}")
            return None
    
    async def get_batch_regime_analysis(
        self, 
        symbols_data: Dict[str, Dict]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get market regime analysis for multiple symbols.
        
        Args:
            symbols_data: Dictionary mapping symbols to their data
            
        Returns:
            Dictionary mapping symbols to their regime analysis
        """
        results = {}
        
        for symbol, data in symbols_data.items():
            try:
                # Extract indicators from the data
                indicators = data.get('indicators', {})
                
                # Get regime analysis
                regime = await self.get_market_regime(symbol, indicators)
                results[symbol] = regime
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.log_error(f"Error analyzing regime for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results
    
    def validate_regime_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate a regime response for quality.
        
        Args:
            response: Regime response to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not response:
            return False
        
        # Check required fields
        required_fields = ['symbol', 'regime', 'factors', 'confidence']
        for field in required_fields:
            if field not in response:
                return False
        
        # Check confidence threshold
        if response.get('confidence', 0) < 0.5:
            logger.log_warning(f"Low confidence regime analysis: {response.get('confidence')}")
            return False
        
        return True

# Utility function for standalone usage
async def analyze_market_regime(
    symbol: str, 
    indicators: Dict[str, any], 
    position_info: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Analyze market regime for a single symbol.
    
    Args:
        symbol: Trading symbol
        indicators: Technical indicators data
        position_info: Current position information
        
    Returns:
        Market regime analysis or None if failed
    """
    async with MarketAdvisor() as advisor:
        return await advisor.get_market_regime(symbol, indicators, position_info)
