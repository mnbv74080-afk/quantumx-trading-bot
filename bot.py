import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
from xgboost import XGBClassifier

# ==========================================
# 1. إعدادات التليجرام الجديدة وحساب المسابقة (100K)
# ==========================================
# قائمة البوتات الجديدة (Bahaa Salah & B S)
TELEGRAM_BOTS = [
    {
        "name": "XAUUSD PRO (Bahaa Salah)",
        "token": "8893812630:AAEQcDzphfrK8_WrkXWHaC8D_81svB0djHQ",
        "chat_id": "5806716179"
    },
    {
        "name": "XAUUSD PRO (B S)",
        "token": "8812826771:AAELi8qVCegL_37t2sjdjeYLO5JMo-V8aKg",
        "chat_id": "7691199088"
    }
]

ACCOUNT_BALANCE = 100000.0  # رأس مال حساب المسابقة

# 🚨 السقف الأعلى المسموح للوت يومياً حسب قوانين المسابقة
MAX_ALLOWED_LOTS = {
    "EURUSD": 5.0,
    "GBPUSD": 5.0,
    "XAUUSD": 3.0,
    "GER30": 3.0,
    "US30": 3.0
}

# ⚙️ تخصيص الاستراتيجيات بفرص مضاعفة ومعدلة للمسابقة (Flexible / High-Frequency Config)
ASSETS_CONFIG = {
    "XAUUSD": {
        "ticker": "GC=F", "period": "5d", "interval": "15m", "higher_interval": "1h",
        "tp1_mult": 1.5, "tp2_mult": 3.5, "min_prob": 0.70, "atr_sl_mult": 1.6, "min_adx": 20,
        "contract_size": 100
    },
    "EURUSD": {
        "ticker": "EURUSD=X", "period": "5d", "interval": "15m", "higher_interval": "1h",
        "tp1_mult": 1.5, "tp2_mult": 3.0, "min_prob": 0.68, "atr_sl_mult": 1.2, "min_adx": 20,
        "contract_size": 100000
    },
    "GBPUSD": {
        "ticker": "GBPUSD=X", "period": "5d", "interval": "15m", "higher_interval": "1h",
        "tp1_mult": 1.5, "tp2_mult": 3.0, "min_prob": 0.68, "atr_sl_mult": 1.2, "min_adx": 20,
        "contract_size": 100000
    },
    "US30": {
        "ticker": "^DJI", "period": "5d", "interval": "15m", "higher_interval": "1h",
        "tp1_mult": 1.5, "tp2_mult": 4.0, "min_prob": 0.72, "atr_sl_mult": 1.8, "min_adx": 22,
        "start_hour": 15, "start_min": 30, "end_hour": 19, "end_min": 0,
        "contract_size": 1
    },
    "GER30": {
        "ticker": "^GDAXI", "period": "5d", "interval": "15m", "higher_interval": "1h",
        "tp1_mult": 1.5, "tp2_mult": 3.5, "min_prob": 0.72, "atr_sl_mult": 1.6, "min_adx": 22,
        "start_hour": 9, "start_min": 0, "end_hour": 12, "end_min": 30,
        "contract_size": 1
    }
}

# ==========================================
# 2. الدوال التنفيذية وإدارة المخاطر
# ==========================================
def send_telegram(message):
    """إرسال التنبيهات إلى كلا البوتين في نفس الوقت"""
    for bot in TELEGRAM_BOTS:
        url = f"https://api.telegram.org/bot{bot['token']}/sendMessage"
        payload = {"chat_id": bot["chat_id"], "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"❌ خطأ تليجرام في {bot['name']}: {e}")

def calculate_dynamic_lot(symbol, sl_dist_price, confidence_prob):
    """حساب اللوت ديناميكياً بحجم مخاطرة 1% - 2% وسقف أقصى للمسابقة"""
    risk_pct = 0.02 if confidence_prob >= 0.80 else 0.01
    risk_amount = ACCOUNT_BALANCE * risk_pct
    
    contract_size = ASSETS_CONFIG[symbol]["contract_size"]
    
    if sl_dist_price <= 0:
        return 0.1
        
    calculated_lot = risk_amount / (sl_dist_price * contract_size)
    
    max_contest_lot = MAX_ALLOWED_LOTS.get(symbol, 5.0)
    lot = max(0.01, min(max_contest_lot, np.round(calculated_lot, 2)))
    return round(lot, 2)

def is_asset_trading_window(symbol):
    now = datetime.now().time()
    config = ASSETS_CONFIG.get(symbol)
    
    if "start_hour" in config:
        start_time = dtime(config["start_hour"], config.get("start_min", 0), 0)
        end_time = dtime(config["end_hour"], config.get("end_min", 0), 0)
        return start_time <= now <= end_time
    
    return (dtime(8, 0, 0) <= now <= dtime(13, 0, 0)) or (dtime(15, 0, 0) <= now <= dtime(20, 0, 0))

# ==========================================
# 3. معالجة البيانات الفنية وXGBoost
# ==========================================
def fetch_and_process_data(symbol):
    config = ASSETS_CONFIG[symbol]
    try:
        df = yf.download(tickers=config["ticker"], period=config["period"], interval=config["interval"], progress=False)
        if df.empty or len(df) < 50:
            return None
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['returns'] = df['close'].pct_change()
        
        # RVOL (السيولة النسبية)
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['rvol'] = df['volume'] / (df['vol_ma'] + 1e-9)
        
        # ATR (مؤشر مدى الحركة الحقيقي)
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        # ADX (مؤشر قوة الاتجاه)
        df['up'] = df['high'] - df['high'].shift(1)
        df['down'] = df['low'].shift(1) - df['low']
        df['+dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
        df['-dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        
        tr_smooth = df['tr'].rolling(14).sum()
        p_di = 100 * (df['+dm'].rolling(14).sum() / (tr_smooth + 1e-9))
        m_di = 100 * (df['-dm'].rolling(14).sum() / (tr_smooth + 1e-9))
        df['adx'] = (100 * (np.abs(p_di - m_di) / (p_di + m_di + 1e-9))).rolling(14).mean()
        
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        return df.dropna()
    except Exception as e:
        print(f"Data Fetch Error for {symbol}: {e}")
        return None

def get_higher_tf_trend(symbol):
    config = ASSETS_CONFIG[symbol]
    try:
        df = yf.download(tickers=config["ticker"], period="5d", interval=config["higher_interval"], progress=False)
        if df.empty: return "NEUTRAL"
        ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
        close_price = df['Close'].iloc[-1]
        return "BULLISH" if close_price > ema20 else "BEARISH"
    except:
        return "NEUTRAL"

def train_xgboost(df):
    features = ['returns', 'ema_20', 'ema_50', 'atr', 'adx', 'rvol']
    X, y = df[features], df['target']
    model = XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.03, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    return model, features

# ==========================================
# 4. المحرك الرئيسي للبطولة
# ==========================================
def main():
    print("⚡ Dual-Bot High-Frequency Tournament Engine Activated (100K Cloud)...")
    send_telegram(
        "⚡ *المحرك المطور للمسابقة - ربط مزدوج (100K)*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🤖 *النظام:* تم ربط البوتين (BAHAA_Trading_bot & BAHAA1_Trading_bot) بنجاح.\n"
        "🎯 *الحساسية:* وضع تكثيف الفرص مفعل لاقتناص أكبر عدد من الصفقات.\n"
        "🛡️ *الالتزام:* حماية رأس المال وسقف اللوت مفعلين بالكامل (3 لوت للمؤشرات والذهب / 5 للعملات).\n"
        "🚀 *حالة النظام:* مراقبة فورية لجميع الأزواج وإرسال الإشارات مجتمعة!"
    )

    scanned_candles = {symbol: None for symbol in ASSETS_CONFIG.keys()}

    while True:
        try:
            for symbol, config in ASSETS_CONFIG.items():
                if not is_asset_trading_window(symbol): continue

                df = fetch_and_process_data(symbol)
                if df is None or len(df) < 50: continue

                current_candle_time = df.index[-1]

                if scanned_candles[symbol] != current_candle_time:
                    higher_trend = get_higher_tf_trend(symbol)
                    model, features = train_xgboost(df)
                    
                    latest = df[features].iloc[[-1]]
                    pred, prob = model.predict(latest)[0], model.predict_proba(latest)[0]
                    
                    current_price = float(df['close'].iloc[-1])
                    atr = float(df['atr'].iloc[-1])
                    adx = float(df['adx'].iloc[-1])
                    rvol = float(df['rvol'].iloc[-1])
                    ema20 = float(df['ema_20'].iloc[-1])

                    sl_dist = atr * config['atr_sl_mult']
                    signal = None

                    # فلترة محسنة لدخول الصفقات
                    if rvol >= 1.0:
                        if pred == 1 and prob[1] >= config['min_prob'] and adx >= config['min_adx'] and higher_trend == "BULLISH" and current_price > ema20:
                            signal = "BUY"
                            confidence = prob[1] * 100
                            entry_price = current_price
                            sl = entry_price - sl_dist
                            tp1 = entry_price + (sl_dist * config['tp1_mult'])
                            tp2 = entry_price + (sl_dist * config['tp2_mult'])

                        elif pred == 0 and prob[0] >= config['min_prob'] and adx >= config['min_adx'] and higher_trend == "BEARISH" and current_price < ema20:
                            signal = "SELL"
                            confidence = prob[0] * 100
                            entry_price = current_price
                            sl = entry_price + sl_dist
                            tp1 = entry_price - (sl_dist * config['tp1_mult'])
                            tp2 = entry_price - (sl_dist * config['tp2_mult'])

                    if signal:
                        recommended_lot = calculate_dynamic_lot(symbol, sl_dist, confidence/100)
                        max_limit_note = f" (الحد الأقصى المسموح {MAX_ALLOWED_LOTS[symbol]} Lot)"
                        risk_flag = "🔥 فرصة عالية القوة (ثقة +80%)" if confidence >= 80 else "🛡️ صفقة قياسية"

                        msg = (
                            f"🏆 *إشارة مسابقة معتمدة ({symbol})*\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"📌 *الأصل:* `{symbol}` | 🚦 *نوع الصفقة:* `{signal}`\n"
                            f"🎯 *الحجم الموصى به:* `{recommended_lot} Lot`{max_limit_note}\n"
                            f"🏷️ *التصنيف:* {risk_flag}\n"
                            f"📍 *سعر الدخول:* `{entry_price:.4f}`\n"
                            f"🛑 *وقف الخسارة (SL):* `{sl:.4f}`\n\n"
                            f"🎯 *الهدف الأول (TP1):* `{tp1:.4f}` (اغلق 50% + حول الستوب للدخول)\n"
                            f"🎯 *الهدف الثاني (TP2):* `{tp2:.4f}` (عائد مضاعف)\n"
                            f"🤖 *دقة الذكاء الاصطناعي:* `{confidence:.1f}%`\n"
                            f"📊 *مؤشر السيولة RVOL:* `{rvol:.2f}`\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"💡 *تعليمات التنفيذ:* نفذ الصفقة بحجم اللوت المحسوب أعلاه مباشرة."
                        )
                        send_telegram(msg)
                        scanned_candles[symbol] = current_candle_time

            time.sleep(15)

        except Exception as e:
            print(f"⚠️ خطأ المحرك: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
