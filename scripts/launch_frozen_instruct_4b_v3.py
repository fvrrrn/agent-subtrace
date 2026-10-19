"""Same authorized five-arm experiment, both serialization boundaries repaired."""
import time
import traceback

import launch_frozen_instruct_4b_v2 as pipeline

pipeline.RUN = pipeline.ROOT / 'runs/frozen-instruct-4b-v3'

if __name__ == '__main__':
    try:
        pipeline.main()
    except Exception:
        failure = {'status': 'failed', 'traceback': traceback.format_exc()}
        pipeline.write(pipeline.RUN / f'failure_{time.time_ns()}.json', failure)
        pipeline.write(pipeline.RUN / 'status.json', failure)
        raise
