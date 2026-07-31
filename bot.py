import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# =====================================================================
# 1. إعدادات البوتين والأصول (Configuration)
# =====================================================================

BOTS_CONFIG = [
    {
        "name": "BAHAA_Trading_bot",
        "token": "8893812630:AAEQcDzphfrK8_WrkXWHaC8D_81svB0djHQ",
        "chat_id": "5806716179"
    },
    {
        "name": "BAHAA1_Trading_bot",
        "token": "8812826771:AAELi8qVCegL_37t2sjdjeYLO5JMo-V8aKg",
        "chat_id": "7691199088"
    }
]

# رموز الأسعار اللحظية الفورية
SYMBOLS_MAP = {
    'XAUUSD': 'GC=F',       # الذهب
    'US30': 'YM=F',         # الداو جونز
    'NAS100': 'NQ=F',       # النازداك
    'GER30': '^GDAXI',      # الداكس
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'GBPJPY': 'GBPJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDCAD': 'USDCAD=X'
}

MIN_AI_ACCURACY = {
    'XAUUSD': 0.70,
    'NAS100': 0.70,
    'US30': 0.70,
    'GER30': 0.70,
    'DEFAULT': 0.75
}

last_signal_time = {}
COOLDOWN_PERIOD = 3600  # زيادة مانع التكرار إلى ساعة (3600 ثانية) لتفادي الصفقات العشوائية

active_trades = []

# =====================================================================
# 2. وظيفة إرسال الرسائل والتنبيهات
# =====================================================================

def broadcast_telegram_message(message):
    for bot in BOTS_CONFIG:
        url = f"https://api.telegram.org/bot{bot['token']}/sendMessage"
        payload = {
            "chat_id": bot['chat_id'],
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                print(f"✅ تم الإرسال عبر: {bot['name']}")
            else:
                print(f"⚠️ فشل الإرسال عبر {bot['name']}: {res.text}")
        except Exception as e:
            print(f"❌ خطأ شبكة: {e}")

# =====================================================================
# 3. محرك جلب البيانات والتحليل المتقدم (مُصحح ومُطور)
# =====================================================================

def fetch_market_data(ticker_symbol):
    """جلب بيانات الشارت بدقة عالية وحساب التغييرات اللحظية"""
    try:
        df = yf.download(ticker_symbol, period="5d", interval="15m", progress=False)
        if df.empty or len(df) < 50:
            return None
        
        # معالجة الفهارس وإزالة الأسماء المتعددة للدرجات
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"⚠️ خطأ جلب بيانات {ticker_symbol}: {e}")
        return None

def analyze_market_advanced(df, symbol):
    """محلل دقيق يمنع الصفقات الفورية عند بدء التشغيل إلا عند توفر شروط صارمة"""
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    # إصلاح حساب RVOL مع بديل مدى الشمعة إذا كان الأحجام مفقودة (كما في المؤشرات)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        vol_mean = df['volume'].rolling(20).mean()
        df['RVOL'] = np.where(vol_mean > 0, df['volume'] / (vol_mean + 1e-9), 1.0)
    else:
        # البديل: مقارنة مدى الشمعة الحالية بمعدل المدى لآخر 20 شمعة
        candle_range = df['high'] - df['low']
        range_mean = candle_range.rolling(20).mean()
        df['RVOL'] = np.where(range_mean > 0, candle_range / (range_mean + 1e-9), 1.0)

    latest = df.iloc[-1]
    current_price = latest['close']
    atr = latest['ATR']
    rvol = float(latest['RVOL'])

    # التحقق من أن السعر حديث وغير صفري
    if pd.isna(current_price) or current_price <= 0:
        return None

    ai_score = 0.40  # تخفيض النقطة المبدئية لمنع التوصيات الفورية
    signal_direction = None

    # شروط صعود قوية (BUY)
    if current_price > latest['EMA_200'] and latest['EMA_20'] > latest['EMA_50']:
        if 45 <= latest['RSI'] <= 65:
            ai_score += 0.20
        if rvol >= 1.2:  # اشتراط سيولة أعلى من المتوسط
            ai_score += 0.20
        if df.iloc[-1]['close'] > df.iloc[-2]['high']:  # كسر القمة السابقة
            ai_score += 0.10
        signal_direction = 'BUY'

    # شروط هبوط قوية (SELL)
    elif current_price < latest['EMA_200'] and latest['EMA_20'] < latest['EMA_50']:
        if 35 <= latest['RSI'] <= 55:
            ai_score += 0.20
        if rvol >= 1.2:
            ai_score += 0.20
        if df.iloc[-1]['close'] < df.iloc[-2]['low']:  # كسر القاع السابق
            ai_score += 0.10
        signal_direction = 'SELL'

    required_accuracy = MIN_AI_ACCURACY.get(symbol, MIN_AI_ACCURACY['DEFAULT'])

    # شرط إرسال التوصية فقط في حال تجاوز الدقة المحددة وحضور سيولة مناسبة
    if signal_direction and ai_score >= required_accuracy and rvol >= 1.1:
        multiplier = 2.0 if symbol in ['XAUUSD', 'NAS100', 'US30', 'GER30'] else 1.5
        decimals = 2 if symbol in ['XAUUSD', 'US30', 'NAS100', 'GER30', 'GBPJPY', 'USDJPY'] else 4

        if signal_direction == 'BUY':
            sl = current_price - (atr * multiplier)
            tp1 = current_price + (atr * 1.5)
            tp2 = current_price + (atr * 3.0)
        else:
            sl = current_price + (atr * multiplier)
            tp1 = current_price - (atr * 1.5)
            tp2 = current_price - (atr * 3.0)

        return {
            'symbol': symbol,
            'direction': signal_direction,
            'price': round(current_price, decimals),
            'sl': round(sl, decimals),
            'tp1': round(tp1, decimals),
            'tp2': round(tp2, decimals),
            'ai_accuracy': round(ai_score * 100, 1),
            'rvol': round(rvol, 2)
        }

    return None

# =====================================================================
# 4. تقرير بدء التشغيل ومتابعة الصفقات
# =====================================================================

def send_startup_report():
    print("🔄 جاري إعداد تقرير الأسعار المباشرة...")
    report_msg = "🚀 **تم تحديث وتشغيل السكربت بنجاح (XAUUSD PRO)**\n\n📊 **الأسعار المباشرة الحالية للأصول:**\n"
    
    for symbol, ticker in SYMBOLS_MAP.items():
        df = fetch_market_data(ticker)
        if df is not None:
            last_price = float(df.iloc[-1]['close'])
            decimals = 2 if symbol in ['XAUUSD', 'US30', 'NAS100', 'GER30', 'GBPJPY', 'USDJPY'] else 4
            report_msg += f"• `{symbol}`: **{round(last_price, decimals)}**\n"
        else:
            report_msg += f"• `{symbol}`: ⚠️ متعذر الجلب حالياً\n"
            
    report_msg += "\n🔍 *جاري الفحص المباشر لاقتناص الفرص ذات السيولة العالية فقط...*"
    broadcast_telegram_message(report_msg)

def track_active_trades():
    global active_trades
    trades_to_remove = []

    for trade in active_trades:
        symbol = trade['symbol']
        ticker = SYMBOLS_MAP[symbol]
        df = fetch_market_data(ticker)
        
        if df is None:
            continue
            
        latest = df.iloc[-1]
        high_price = float(latest['high'])
        low_price = float(latest['low'])

        if trade['direction'] == 'BUY':
            if low_price <= trade['sl']:
                msg = f"🛑 **تحديث صفقة {symbol} (BUY)**\n\nتم ضرب وقف الخسارة (SL) عند `{trade['sl']}`."
                broadcast_telegram_message(msg)
                trades_to_remove.append(trade)
                continue

            if high_price >= trade['tp2']:
                msg = f"🚀🚀 **تحديث صفقة {symbol} (BUY)**\n\n🎯 **تم تحقيق Target 2 بنجاح عند `{trade['tp2']}`!** 🔥"
                broadcast_telegram_message(msg)
                trades_to_remove.append(trade)
                continue

            if high_price >= trade['tp1'] and not trade['tp1_hit']:
                trade['tp1_hit'] = True
                msg = f"🎯 **تحديث صفقة {symbol} (BUY)**\n\n✅ **تم تحقيق Target 1 عند `{trade['tp1']}`!**\n💡 يُنصح بنقل الستوب لنقطة الدخول `{trade['entry']}`."
                broadcast_telegram_message(msg)

        elif trade['direction'] == 'SELL':
            if high_price >= trade['sl']:
                msg = f"🛑 **تحديث صفقة {symbol} (SELL)**\n\nتم ضرب وقف الخسارة (SL) عند `{trade['sl']}`."
                broadcast_telegram_message(msg)
                trades_to_remove.append(trade)
                continue

            if low_price <= trade['tp2']:
                msg = f"🚀🚀 **تحديث صفقة {symbol} (SELL)**\n\n🎯 **تم تحقيق Target 2 بنجاح عند `{trade['tp2']}`!** 🔥"
                broadcast_telegram_message(msg)
                trades_to_remove.append(trade)
                continue

            if low_price <= trade['tp1'] and not trade['tp1_hit']:
                trade['tp1_hit'] = True
                msg = f"🎯 **تحديث صفقة {symbol} (SELL)**\n\n✅ **تم تحقيق Target 1 عند `{trade['tp1']}`!**\n💡 يُنصح بنقل الستوب لنقطة الدخول `{trade['entry']}`."
                broadcast_telegram_message(msg)

    for trade in trades_to_remove:
        active_trades.remove(trade)

# =====================================================================
# 5. حلقة التنفيذ (Execution Loop)
# =====================================================================

def is_cooldown_active(symbol):
    current_time = time.time()
    if symbol in last_signal_time:
        if current_time - last_signal_time[symbol] < COOLDOWN_PERIOD:
            return True
    return False

def format_telegram_alert(signal):
    direction_emoji = "🟢 BUY" if signal['direction'] == 'BUY' else "🔴 SELL"
    return f"""
🚨 **تنبيه فرصة تداول جديدة (XAUUSD PRO)**

📌 **الأصل:** `{signal['symbol']}` | **الاتجاه:** {direction_emoji}
🎯 **سعر الدخول:** `{signal['price']}`
🛑 **وقف الخسارة (SL):** `{signal['sl']}`

🔹 **الهدف الأول (TP1):** `{signal['tp1']}`
🔹 **الهدف الثاني (TP2):** `{signal['tp2']}`

🤖 **دقة النموذج:** `{signal['ai_accuracy']}%`
📊 **مؤشر السيولة (RVOL):** `{signal['rvol']}`
"""

def run_scanner():
    track_active_trades()
    
    for symbol, ticker in SYMBOLS_MAP.items():
        if is_cooldown_active(symbol):
            continue

        df = fetch_market_data(ticker)
        if df is not None:
            signal = analyze_market_advanced(df, symbol)
            if signal:
                alert_msg = format_telegram_alert(signal)
                broadcast_telegram_message(alert_msg)
                last_signal_time[symbol] = time.time()
                
                active_trades.append({
                    'symbol': symbol,
                    'direction': signal['direction'],
                    'entry': signal['price'],
                    'sl': signal['sl'],
                    'tp1': signal['tp1'],
                    'tp2': signal['tp2'],
                    'tp1_hit': False
                })

def main():
    print("🤖 جاري تشغيل السكربت المعادل المطور...")
    send_startup_report()
    
    while True:
        try:
            run_scanner()
            time.sleep(60)
        except Exception as e:
            print(f"⚠️ حدث خطأ: {e}. محاولة مجدداً خلال 15 ثانية...")
            time.sleep(15)

if __name__ == "__main__":
    main()
