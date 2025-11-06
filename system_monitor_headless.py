#!/usr/bin/env python3
"""
실시간 시스템 리소스 모니터링 시스템 (Headless 버전)
GUI 없이 백그라운드에서 실행되며, 5분 후 PDF 리포트를 생성합니다.
"""

import psutil
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
import time
import datetime
from collections import deque
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import os
import sys


class SystemMonitorHeadless:
    def __init__(self, duration=300, update_interval=1):
        """
        시스템 모니터 초기화 (Headless 모드)

        Args:
            duration: 모니터링 시간 (초), 기본값 300초 (5분)
            update_interval: 업데이트 간격 (초), 기본값 1초
        """
        self.duration = duration
        self.update_interval = update_interval
        self.max_points = duration // update_interval

        # 데이터 저장을 위한 deque 초기화
        self.times = deque(maxlen=self.max_points)
        self.cpu_data = deque(maxlen=self.max_points)
        self.memory_data = deque(maxlen=self.max_points)
        self.disk_read_data = deque(maxlen=self.max_points)
        self.disk_write_data = deque(maxlen=self.max_points)
        self.net_sent_data = deque(maxlen=self.max_points)
        self.net_recv_data = deque(maxlen=self.max_points)
        self.temp_data = deque(maxlen=self.max_points)
        self.cpu_cores_history = []

        # 이전 네트워크/디스크 카운터 저장
        self.prev_net = psutil.net_io_counters()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_time = time.time()

        # 시작 시간
        self.start_time = time.time()

    def get_temperature(self):
        """시스템 온도 조회 (가능한 경우)"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # 사용 가능한 첫 번째 온도 센서 사용
                for name, entries in temps.items():
                    if entries:
                        return entries[0].current
            return None
        except (AttributeError, OSError):
            return None

    def get_system_info(self):
        """시스템 정보 수집"""
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        info = {
            'CPU 코어 수': cpu_count,
            'CPU 주파수': f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A",
            '총 메모리': f"{mem.total / (1024**3):.2f} GB",
            '사용 메모리': f"{mem.used / (1024**3):.2f} GB",
            '총 디스크': f"{disk.total / (1024**3):.2f} GB",
            '사용 디스크': f"{disk.used / (1024**3):.2f} GB",
        }

        return info

    def update_data(self):
        """시스템 데이터 수집 및 업데이트"""
        current_time = time.time() - self.start_time
        self.times.append(current_time)

        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_data.append(cpu_percent)

        # 코어별 CPU
        cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
        self.cpu_cores_history.append(cpu_cores)

        # 메모리 사용률
        memory = psutil.virtual_memory()
        self.memory_data.append(memory.percent)

        # 디스크 I/O
        current_disk = psutil.disk_io_counters()
        current_time_stamp = time.time()
        time_delta = current_time_stamp - self.prev_time

        if current_disk and self.prev_disk and time_delta > 0:
            disk_read_rate = (current_disk.read_bytes - self.prev_disk.read_bytes) / time_delta / (1024**2)
            disk_write_rate = (current_disk.write_bytes - self.prev_disk.write_bytes) / time_delta / (1024**2)
        else:
            disk_read_rate = 0
            disk_write_rate = 0

        self.disk_read_data.append(disk_read_rate)
        self.disk_write_data.append(disk_write_rate)

        # 네트워크 트래픽
        current_net = psutil.net_io_counters()

        if current_net and self.prev_net and time_delta > 0:
            net_sent_rate = (current_net.bytes_sent - self.prev_net.bytes_sent) / time_delta / (1024**2)
            net_recv_rate = (current_net.bytes_recv - self.prev_net.bytes_recv) / time_delta / (1024**2)
        else:
            net_sent_rate = 0
            net_recv_rate = 0

        self.net_sent_data.append(net_sent_rate)
        self.net_recv_data.append(net_recv_rate)

        # 온도
        temp = self.get_temperature()
        self.temp_data.append(temp if temp else 0)

        # 이전 값 업데이트
        if current_disk:
            self.prev_disk = current_disk
        if current_net:
            self.prev_net = current_net
        self.prev_time = current_time_stamp

    def start(self):
        """모니터링 시작 (Headless 모드)"""
        print(f"시스템 모니터링을 시작합니다. (총 {self.duration}초)")
        print("백그라운드에서 데이터를 수집합니다...")
        print()

        # 초기 데이터 수집
        psutil.cpu_percent(interval=None)
        time.sleep(1)

        # 진행률 표시
        total_steps = self.duration // self.update_interval
        step = 0

        start_time = time.time()
        next_update = start_time + self.update_interval

        while time.time() - start_time < self.duration:
            current_time = time.time()

            if current_time >= next_update:
                self.update_data()
                step += 1

                # 진행률 표시 (10% 단위)
                progress = (step / total_steps) * 100
                if step % (total_steps // 10) == 0 or step == total_steps:
                    elapsed = current_time - start_time
                    remaining = self.duration - elapsed
                    print(f"진행률: {progress:.1f}% | "
                          f"경과: {int(elapsed)}초 | "
                          f"남은 시간: {int(remaining)}초 | "
                          f"CPU: {self.cpu_data[-1]:.1f}% | "
                          f"메모리: {self.memory_data[-1]:.1f}%")

                next_update += self.update_interval

            # CPU 사용을 줄이기 위해 짧은 sleep
            time.sleep(0.1)

        print()
        print("모니터링이 완료되었습니다.")
        return self.get_statistics()

    def get_statistics(self):
        """수집된 데이터의 통계 계산"""
        stats = {
            'cpu': {
                'avg': sum(self.cpu_data) / len(self.cpu_data) if self.cpu_data else 0,
                'max': max(self.cpu_data) if self.cpu_data else 0,
                'min': min(self.cpu_data) if self.cpu_data else 0,
            },
            'memory': {
                'avg': sum(self.memory_data) / len(self.memory_data) if self.memory_data else 0,
                'max': max(self.memory_data) if self.memory_data else 0,
                'min': min(self.memory_data) if self.memory_data else 0,
            },
            'disk_read': {
                'avg': sum(self.disk_read_data) / len(self.disk_read_data) if self.disk_read_data else 0,
                'max': max(self.disk_read_data) if self.disk_read_data else 0,
                'total': sum(self.disk_read_data) * self.update_interval if self.disk_read_data else 0,
            },
            'disk_write': {
                'avg': sum(self.disk_write_data) / len(self.disk_write_data) if self.disk_write_data else 0,
                'max': max(self.disk_write_data) if self.disk_write_data else 0,
                'total': sum(self.disk_write_data) * self.update_interval if self.disk_write_data else 0,
            },
            'net_sent': {
                'avg': sum(self.net_sent_data) / len(self.net_sent_data) if self.net_sent_data else 0,
                'max': max(self.net_sent_data) if self.net_sent_data else 0,
                'total': sum(self.net_sent_data) * self.update_interval if self.net_sent_data else 0,
            },
            'net_recv': {
                'avg': sum(self.net_recv_data) / len(self.net_recv_data) if self.net_recv_data else 0,
                'max': max(self.net_recv_data) if self.net_recv_data else 0,
                'total': sum(self.net_recv_data) * self.update_interval if self.net_recv_data else 0,
            },
            'temp': {
                'avg': sum(t for t in self.temp_data if t) / len([t for t in self.temp_data if t]) if any(self.temp_data) else 0,
                'max': max(t for t in self.temp_data if t) if any(self.temp_data) else 0,
                'min': min(t for t in self.temp_data if t) if any(self.temp_data) else 0,
            }
        }
        return stats

    def generate_pdf_report(self, filename='system_monitoring_report.pdf'):
        """PDF 리포트 생성"""
        print(f"PDF 리포트를 생성합니다: {filename}")

        # PDF 문서 설정
        doc = SimpleDocTemplate(filename, pagesize=landscape(A4))
        elements = []
        styles = getSampleStyleSheet()

        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # 소제목 스타일
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        # 제목
        title = Paragraph('시스템 리소스 모니터링 리포트', title_style)
        elements.append(title)

        # 모니터링 정보
        monitoring_info = f"""
        <para alignment='center'>
        <b>모니터링 기간:</b> {self.duration}초 ({self.duration // 60}분)<br/>
        <b>생성 시간:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>데이터 포인트:</b> {len(self.times)}개<br/>
        <b>샘플링 간격:</b> {self.update_interval}초
        </para>
        """
        elements.append(Paragraph(monitoring_info, styles['Normal']))
        elements.append(Spacer(1, 20))

        # 통계 데이터
        stats = self.get_statistics()

        # 통계 테이블
        elements.append(Paragraph('1. 리소스 사용률 통계', subtitle_style))

        stats_data = [
            ['리소스', '평균', '최대', '최소', '단위'],
            ['CPU 사용률', f"{stats['cpu']['avg']:.2f}", f"{stats['cpu']['max']:.2f}",
             f"{stats['cpu']['min']:.2f}", '%'],
            ['메모리 사용률', f"{stats['memory']['avg']:.2f}", f"{stats['memory']['max']:.2f}",
             f"{stats['memory']['min']:.2f}", '%'],
            ['디스크 읽기', f"{stats['disk_read']['avg']:.3f}", f"{stats['disk_read']['max']:.3f}",
             '-', 'MB/s'],
            ['디스크 쓰기', f"{stats['disk_write']['avg']:.3f}", f"{stats['disk_write']['max']:.3f}",
             '-', 'MB/s'],
            ['네트워크 송신', f"{stats['net_sent']['avg']:.3f}", f"{stats['net_sent']['max']:.3f}",
             '-', 'MB/s'],
            ['네트워크 수신', f"{stats['net_recv']['avg']:.3f}", f"{stats['net_recv']['max']:.3f}",
             '-', 'MB/s'],
        ]

        if stats['temp']['avg'] > 0:
            stats_data.append(['시스템 온도', f"{stats['temp']['avg']:.2f}",
                             f"{stats['temp']['max']:.2f}", f"{stats['temp']['min']:.2f}", '°C'])

        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        elements.append(stats_table)
        elements.append(Spacer(1, 20))

        # 누적 데이터 테이블
        elements.append(Paragraph('2. 누적 데이터 전송량', subtitle_style))

        total_data = [
            ['항목', '총량', '단위'],
            ['총 디스크 읽기', f"{stats['disk_read']['total']:.2f}", 'MB'],
            ['총 디스크 쓰기', f"{stats['disk_write']['total']:.2f}", 'MB'],
            ['총 네트워크 송신', f"{stats['net_sent']['total']:.2f}", 'MB'],
            ['총 네트워크 수신', f"{stats['net_recv']['total']:.2f}", 'MB'],
        ]

        total_table = Table(total_data, colWidths=[3*inch, 2*inch, 2*inch])
        total_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        elements.append(total_table)
        elements.append(PageBreak())

        # 그래프 생성 및 추가
        print("그래프를 생성하는 중...")
        self.create_report_graphs()

        elements.append(Paragraph('3. 시각화 그래프', subtitle_style))

        # 그래프 이미지 추가
        graph_files = [
            ('cpu_graph.png', 'CPU 사용률 추이'),
            ('memory_graph.png', '메모리 사용률 추이'),
            ('disk_graph.png', '디스크 I/O 추이'),
            ('network_graph.png', '네트워크 트래픽 추이'),
            ('cores_graph.png', '코어별 평균 CPU 사용률')
        ]

        for graph_file, title in graph_files:
            if os.path.exists(graph_file):
                elements.append(Paragraph(title, styles['Heading3']))
                img = Image(graph_file, width=7*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 15))

        # 시스템 정보
        elements.append(PageBreak())
        elements.append(Paragraph('4. 시스템 정보', subtitle_style))

        info = self.get_system_info()
        info_data = [['항목', '정보']]
        for key, value in info.items():
            info_data.append([key, str(value)])

        info_table = Table(info_data, colWidths=[3*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        elements.append(info_table)

        # PDF 생성
        print("PDF 파일을 저장하는 중...")
        doc.build(elements)

        # 임시 그래프 파일 삭제
        print("임시 파일을 정리하는 중...")
        for graph_file, _ in graph_files:
            if os.path.exists(graph_file):
                os.remove(graph_file)

        print(f"PDF 리포트가 생성되었습니다: {filename}")

    def create_report_graphs(self):
        """리포트용 개별 그래프 생성"""
        times_list = list(self.times)

        plt.style.use('seaborn-v0_8-darkgrid')

        # CPU 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(times_list, list(self.cpu_data), 'r-', linewidth=2, label='CPU 사용률')
        plt.fill_between(times_list, list(self.cpu_data), alpha=0.3, color='red')
        plt.title('CPU 사용률 추이', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('사용률 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('cpu_graph.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 메모리 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(times_list, list(self.memory_data), 'b-', linewidth=2, label='메모리 사용률')
        plt.fill_between(times_list, list(self.memory_data), alpha=0.3, color='blue')
        plt.title('메모리 사용률 추이', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('사용률 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('memory_graph.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 디스크 I/O 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(times_list, list(self.disk_read_data), 'g-', linewidth=2, label='읽기')
        plt.plot(times_list, list(self.disk_write_data), 'orange', linewidth=2, label='쓰기')
        plt.fill_between(times_list, list(self.disk_read_data), alpha=0.3, color='green')
        plt.fill_between(times_list, list(self.disk_write_data), alpha=0.3, color='orange')
        plt.title('디스크 I/O 추이', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('속도 (MB/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('disk_graph.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 네트워크 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(times_list, list(self.net_sent_data), 'm-', linewidth=2, label='송신')
        plt.plot(times_list, list(self.net_recv_data), 'c-', linewidth=2, label='수신')
        plt.fill_between(times_list, list(self.net_sent_data), alpha=0.3, color='magenta')
        plt.fill_between(times_list, list(self.net_recv_data), alpha=0.3, color='cyan')
        plt.title('네트워크 트래픽 추이', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('속도 (MB/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('network_graph.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 코어별 평균 CPU 사용률
        if self.cpu_cores_history:
            # 각 코어의 평균 계산
            num_cores = len(self.cpu_cores_history[0])
            avg_per_core = []
            for core_idx in range(num_cores):
                core_values = [snapshot[core_idx] for snapshot in self.cpu_cores_history]
                avg_per_core.append(sum(core_values) / len(core_values))

            plt.figure(figsize=(12, 6))
            cores = list(range(num_cores))
            bars = plt.bar(cores, avg_per_core, color='steelblue', alpha=0.7, edgecolor='black')

            # 각 바에 값 표시
            for bar, value in zip(bars, avg_per_core):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.title('코어별 평균 CPU 사용률', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('CPU 코어', fontsize=12)
            plt.ylabel('평균 사용률 (%)', fontsize=12)
            plt.ylim(0, max(avg_per_core) * 1.2 if avg_per_core else 100)
            plt.xticks(cores, [f'Core {i}' for i in cores])
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('cores_graph.png', dpi=150, bbox_inches='tight')
            plt.close()


def main():
    """메인 함수"""
    print("=" * 70)
    print("실시간 시스템 리소스 모니터링 시스템 (Headless 모드)")
    print("=" * 70)
    print()

    # 모니터 생성 (5분 = 300초)
    monitor = SystemMonitorHeadless(duration=300, update_interval=1)

    # 모니터링 시작
    stats = monitor.start()

    # PDF 리포트 생성
    print()
    monitor.generate_pdf_report('system_monitoring_report.pdf')

    print()
    print("=" * 70)
    print("모니터링 완료!")
    print("=" * 70)
    print(f"평균 CPU 사용률: {stats['cpu']['avg']:.2f}%")
    print(f"최대 CPU 사용률: {stats['cpu']['max']:.2f}%")
    print(f"평균 메모리 사용률: {stats['memory']['avg']:.2f}%")
    print(f"최대 메모리 사용률: {stats['memory']['max']:.2f}%")
    print(f"총 디스크 읽기: {stats['disk_read']['total']:.2f} MB")
    print(f"총 디스크 쓰기: {stats['disk_write']['total']:.2f} MB")
    print(f"총 네트워크 송신: {stats['net_sent']['total']:.2f} MB")
    print(f"총 네트워크 수신: {stats['net_recv']['total']:.2f} MB")
    if stats['temp']['avg'] > 0:
        print(f"평균 시스템 온도: {stats['temp']['avg']:.2f}°C")
    print()
    print(f"PDF 리포트: system_monitoring_report.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
