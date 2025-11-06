#!/usr/bin/env python3
"""
시스템 모니터링 테스트 (30초)
"""

import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
from collections import deque
import os


def test_monitoring():
    """30초 간단 테스트"""
    print("=" * 60)
    print("시스템 모니터링 테스트 시작 (30초)")
    print("=" * 60)
    print()

    duration = 30
    interval = 1

    times = []
    cpu_data = []
    memory_data = []

    # 초기화
    psutil.cpu_percent(interval=None)
    time.sleep(1)

    start_time = time.time()

    for i in range(duration):
        elapsed = time.time() - start_time
        times.append(elapsed)

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        cpu_data.append(cpu)
        memory_data.append(mem)

        if i % 5 == 0:
            print(f"[{i:2d}초] CPU: {cpu:5.1f}% | 메모리: {mem:5.1f}%")

        time.sleep(interval)

    print()
    print("데이터 수집 완료. 그래프 생성 중...")

    # 간단한 그래프 생성
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(times, cpu_data, 'r-', linewidth=2)
    plt.fill_between(times, cpu_data, alpha=0.3, color='red')
    plt.title('CPU 사용률', fontsize=14, fontweight='bold')
    plt.xlabel('시간 (초)')
    plt.ylabel('%')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(times, memory_data, 'b-', linewidth=2)
    plt.fill_between(times, memory_data, alpha=0.3, color='blue')
    plt.title('메모리 사용률', fontsize=14, fontweight='bold')
    plt.xlabel('시간 (초)')
    plt.ylabel('%')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_graph.png', dpi=150)
    plt.close()

    print("테스트 그래프 생성 완료: test_graph.png")
    print()
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"평균 CPU: {sum(cpu_data)/len(cpu_data):.2f}%")
    print(f"평균 메모리: {sum(memory_data)/len(memory_data):.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    test_monitoring()
