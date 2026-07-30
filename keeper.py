import subprocess
import time
import sys

def run_bot():
    """تشغيل البوت ومراقبته وإعادة تشغيله فوراً عند حدوث أي خطأ أو توقف"""
    while True:
        print("🚀 [Keeper] جاري تشغيل التداول المباشر...")
        try:
            # تشغيل bot.py وانتظار إخراجه
            process = subprocess.Popen([sys.executable, "bot.py"])
            process.wait()
            
            # إذا انتهت العملية لأي سبب، سيصل الكود هنا
            print(f"⚠️ [Keeper] توقف البوت بكود الخروج: {process.returncode}. إعادة التشغيل خلال 5 ثوانٍ...")
        except Exception as e:
            print(f"❌ [Keeper] خطأ أثناء التشغيل: {e}")
        
        time.sleep(5)  # الانتظار 5 ثوانٍ قبل إعادة التشغيل لتجنب الضغط العالي

if __name__ == "__main__":
    run_bot()
