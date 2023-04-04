import schedule
import time

def job(name):
    print(f'Doing {name}s job')

schedule.every(2).seconds.do(job, name='Tomas')

while True:
    schedule.run_pending()
    time.sleep(1)