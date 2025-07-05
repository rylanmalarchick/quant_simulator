from apscheduler.schedulers.blocking import BlockingScheduler
from src.signal.generate import generate_signals

def start_scheduler():
    """
    Starts the APScheduler.
    """
    scheduler = BlockingScheduler()
    
    # Schedule signal generation
    scheduler.add_job(generate_signals, 'cron', hour='10,11,13,14', minute='0,30', second='0')
    
    print("Scheduler started. Press Ctrl+C to exit.")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == '__main__':
    start_scheduler()
