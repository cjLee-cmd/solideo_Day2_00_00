#!/usr/bin/env python3
"""
실시간 시스템 리소스 모니터링 시스템
CPU, 메모리, 디스크, 네트워크, 온도 등 시스템 리소스를 실시간으로 모니터링하고 시각화합니다.
"""

import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time
import datetime
from collections import deque
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import sys


class SystemMonitor:
    def __init__(self, duration=300, update_interval=1):
        """
        시스템 모니터 초기화

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

        # 이전 네트워크/디스크 카운터 저장
        self.prev_net = psutil.net_io_counters()
        self.prev_disk = psutil.disk_io_counters()
        self.prev_time = time.time()

        # 시작 시간
        self.start_time = time.time()

        # 그래프 설정
        self.setup_plots()

    def setup_plots(self):
        """실시간 그래프 설정"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('실시간 시스템 리소스 모니터링', fontsize=16, fontweight='bold')

        # GridSpec을 사용하여 레이아웃 구성
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.4, wspace=0.3)

        # 각 서브플롯 생성
        self.ax_cpu = self.fig.add_subplot(gs[0, 0])
        self.ax_memory = self.fig.add_subplot(gs[0, 1])
        self.ax_disk_io = self.fig.add_subplot(gs[0, 2])
        self.ax_network = self.fig.add_subplot(gs[1, 0])
        self.ax_temp = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_cpu_cores = self.fig.add_subplot(gs[2, :])

        # 축 설정
        self.ax_cpu.set_title('CPU 사용률 (%)', fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)

        self.ax_memory.set_title('메모리 사용률 (%)', fontweight='bold')
        self.ax_memory.set_ylim(0, 100)

        self.ax_disk_io.set_title('디스크 I/O (MB/s)', fontweight='bold')

        self.ax_network.set_title('네트워크 트래픽 (MB/s)', fontweight='bold')

        self.ax_temp.set_title('시스템 온도 (°C)', fontweight='bold')

        self.ax_info.set_title('시스템 정보', fontweight='bold')
        self.ax_info.axis('off')

        self.ax_cpu_cores.set_title('코어별 CPU 사용률 (%)', fontweight='bold')
        self.ax_cpu_cores.set_ylim(0, 100)

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
            '총 디스크': f"{disk.total / (1024**3):.2f} GB",
            '운영체제': f"{psutil.LINUX if hasattr(psutil, 'LINUX') else 'Unknown'}",
        }

        return info

    def update_data(self):
        """시스템 데이터 수집 및 업데이트"""
        current_time = time.time() - self.start_time
        self.times.append(current_time)

        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_data.append(cpu_percent)

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

    def animate(self, frame):
        """애니메이션 프레임 업데이트"""
        # 데이터 업데이트
        self.update_data()

        # 모든 그래프 클리어
        self.ax_cpu.clear()
        self.ax_memory.clear()
        self.ax_disk_io.clear()
        self.ax_network.clear()
        self.ax_temp.clear()
        self.ax_info.clear()
        self.ax_cpu_cores.clear()

        # 시간 축
        times_list = list(self.times)

        # CPU 그래프
        self.ax_cpu.plot(times_list, list(self.cpu_data), 'r-', linewidth=2, label='CPU')
        self.ax_cpu.fill_between(times_list, list(self.cpu_data), alpha=0.3, color='red')
        self.ax_cpu.set_title('CPU 사용률 (%)', fontweight='bold')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_xlabel('시간 (초)')
        self.ax_cpu.set_ylabel('%')
        self.ax_cpu.grid(True, alpha=0.3)
        self.ax_cpu.legend(loc='upper right')

        # 메모리 그래프
        self.ax_memory.plot(times_list, list(self.memory_data), 'b-', linewidth=2, label='메모리')
        self.ax_memory.fill_between(times_list, list(self.memory_data), alpha=0.3, color='blue')
        self.ax_memory.set_title('메모리 사용률 (%)', fontweight='bold')
        self.ax_memory.set_ylim(0, 100)
        self.ax_memory.set_xlabel('시간 (초)')
        self.ax_memory.set_ylabel('%')
        self.ax_memory.grid(True, alpha=0.3)
        self.ax_memory.legend(loc='upper right')

        # 디스크 I/O 그래프
        self.ax_disk_io.plot(times_list, list(self.disk_read_data), 'g-', linewidth=2, label='읽기')
        self.ax_disk_io.plot(times_list, list(self.disk_write_data), 'orange', linewidth=2, label='쓰기')
        self.ax_disk_io.fill_between(times_list, list(self.disk_read_data), alpha=0.3, color='green')
        self.ax_disk_io.fill_between(times_list, list(self.disk_write_data), alpha=0.3, color='orange')
        self.ax_disk_io.set_title('디스크 I/O (MB/s)', fontweight='bold')
        self.ax_disk_io.set_xlabel('시간 (초)')
        self.ax_disk_io.set_ylabel('MB/s')
        self.ax_disk_io.grid(True, alpha=0.3)
        self.ax_disk_io.legend(loc='upper right')

        # 네트워크 그래프
        self.ax_network.plot(times_list, list(self.net_sent_data), 'm-', linewidth=2, label='송신')
        self.ax_network.plot(times_list, list(self.net_recv_data), 'c-', linewidth=2, label='수신')
        self.ax_network.fill_between(times_list, list(self.net_sent_data), alpha=0.3, color='magenta')
        self.ax_network.fill_between(times_list, list(self.net_recv_data), alpha=0.3, color='cyan')
        self.ax_network.set_title('네트워크 트래픽 (MB/s)', fontweight='bold')
        self.ax_network.set_xlabel('시간 (초)')
        self.ax_network.set_ylabel('MB/s')
        self.ax_network.grid(True, alpha=0.3)
        self.ax_network.legend(loc='upper right')

        # 온도 그래프
        if any(self.temp_data):
            self.ax_temp.plot(times_list, list(self.temp_data), 'y-', linewidth=2, label='온도')
            self.ax_temp.fill_between(times_list, list(self.temp_data), alpha=0.3, color='yellow')
            self.ax_temp.set_xlabel('시간 (초)')
            self.ax_temp.set_ylabel('°C')
            self.ax_temp.legend(loc='upper right')
        else:
            self.ax_temp.text(0.5, 0.5, '온도 센서 없음',
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=self.ax_temp.transAxes,
                            fontsize=12)
        self.ax_temp.set_title('시스템 온도 (°C)', fontweight='bold')
        self.ax_temp.grid(True, alpha=0.3)

        # 시스템 정보
        self.ax_info.axis('off')
        info = self.get_system_info()
        info_text = '\n'.join([f'{k}: {v}' for k, v in info.items()])

        # 현재 상태 추가
        if self.cpu_data:
            info_text += f"\n\n현재 CPU: {self.cpu_data[-1]:.1f}%"
        if self.memory_data:
            info_text += f"\n현재 메모리: {self.memory_data[-1]:.1f}%"
        if self.temp_data and self.temp_data[-1]:
            info_text += f"\n현재 온도: {self.temp_data[-1]:.1f}°C"

        elapsed = time.time() - self.start_time
        remaining = max(0, self.duration - elapsed)
        info_text += f"\n\n경과 시간: {int(elapsed)}초"
        info_text += f"\n남은 시간: {int(remaining)}초"

        self.ax_info.text(0.1, 0.9, info_text,
                         verticalalignment='top',
                         fontsize=10,
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 코어별 CPU 사용률
        cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cores = list(range(len(cpu_percent_per_core)))
        bars = self.ax_cpu_cores.bar(cores, cpu_percent_per_core, color='steelblue', alpha=0.7)

        # 각 바에 값 표시
        for bar, value in zip(bars, cpu_percent_per_core):
            height = bar.get_height()
            self.ax_cpu_cores.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{value:.1f}%',
                                  ha='center', va='bottom', fontsize=8)

        self.ax_cpu_cores.set_title('코어별 CPU 사용률 (%)', fontweight='bold')
        self.ax_cpu_cores.set_xlabel('CPU 코어')
        self.ax_cpu_cores.set_ylabel('%')
        self.ax_cpu_cores.set_ylim(0, 100)
        self.ax_cpu_cores.set_xticks(cores)
        self.ax_cpu_cores.set_xticklabels([f'Core {i}' for i in cores])
        self.ax_cpu_cores.grid(True, alpha=0.3, axis='y')

        # 그래프 간격 조정
        self.fig.tight_layout()

        # 모니터링 종료 체크
        if time.time() - self.start_time >= self.duration:
            print("\n모니터링이 완료되었습니다. PDF 리포트를 생성합니다...")
            plt.close()
            return

    def start(self):
        """모니터링 시작"""
        print(f"시스템 모니터링을 시작합니다. (총 {self.duration}초)")
        print("실시간 그래프 창이 열립니다...")

        # 초기 데이터 수집
        psutil.cpu_percent(interval=None)
        time.sleep(1)

        # 애니메이션 생성 및 실행
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.update_interval * 1000,
            cache_frame_data=False
        )

        plt.show()

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
                'total': sum(self.disk_read_data) if self.disk_read_data else 0,
            },
            'disk_write': {
                'avg': sum(self.disk_write_data) / len(self.disk_write_data) if self.disk_write_data else 0,
                'max': max(self.disk_write_data) if self.disk_write_data else 0,
                'total': sum(self.disk_write_data) if self.disk_write_data else 0,
            },
            'net_sent': {
                'avg': sum(self.net_sent_data) / len(self.net_sent_data) if self.net_sent_data else 0,
                'max': max(self.net_sent_data) if self.net_sent_data else 0,
                'total': sum(self.net_sent_data) if self.net_sent_data else 0,
            },
            'net_recv': {
                'avg': sum(self.net_recv_data) / len(self.net_recv_data) if self.net_recv_data else 0,
                'max': max(self.net_recv_data) if self.net_recv_data else 0,
                'total': sum(self.net_recv_data) if self.net_recv_data else 0,
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
        <b>데이터 포인트:</b> {len(self.times)}개
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
            ['디스크 읽기', f"{stats['disk_read']['avg']:.2f}", f"{stats['disk_read']['max']:.2f}",
             '-', 'MB/s'],
            ['디스크 쓰기', f"{stats['disk_write']['avg']:.2f}", f"{stats['disk_write']['max']:.2f}",
             '-', 'MB/s'],
            ['네트워크 송신', f"{stats['net_sent']['avg']:.2f}", f"{stats['net_sent']['max']:.2f}",
             '-', 'MB/s'],
            ['네트워크 수신', f"{stats['net_recv']['avg']:.2f}", f"{stats['net_recv']['max']:.2f}",
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
        self.create_report_graphs()

        elements.append(Paragraph('3. 시각화 그래프', subtitle_style))

        # 그래프 이미지 추가
        graph_files = ['cpu_graph.png', 'memory_graph.png', 'disk_graph.png',
                       'network_graph.png', 'cores_graph.png']

        for graph_file in graph_files:
            if os.path.exists(graph_file):
                img = Image(graph_file, width=7*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 10))

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
        doc.build(elements)

        # 임시 그래프 파일 삭제
        for graph_file in graph_files:
            if os.path.exists(graph_file):
                os.remove(graph_file)

        print(f"PDF 리포트가 생성되었습니다: {filename}")

    def create_report_graphs(self):
        """리포트용 개별 그래프 생성"""
        times_list = list(self.times)

        # CPU 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(times_list, list(self.cpu_data), 'r-', linewidth=2, label='CPU 사용률')
        plt.fill_between(times_list, list(self.cpu_data), alpha=0.3, color='red')
        plt.title('CPU 사용률 추이', fontsize=14, fontweight='bold')
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('사용률 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('cpu_graph.png', dpi=150)
        plt.close()

        # 메모리 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(times_list, list(self.memory_data), 'b-', linewidth=2, label='메모리 사용률')
        plt.fill_between(times_list, list(self.memory_data), alpha=0.3, color='blue')
        plt.title('메모리 사용률 추이', fontsize=14, fontweight='bold')
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('사용률 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('memory_graph.png', dpi=150)
        plt.close()

        # 디스크 I/O 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(times_list, list(self.disk_read_data), 'g-', linewidth=2, label='읽기')
        plt.plot(times_list, list(self.disk_write_data), 'orange', linewidth=2, label='쓰기')
        plt.fill_between(times_list, list(self.disk_read_data), alpha=0.3, color='green')
        plt.fill_between(times_list, list(self.disk_write_data), alpha=0.3, color='orange')
        plt.title('디스크 I/O 추이', fontsize=14, fontweight='bold')
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('속도 (MB/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('disk_graph.png', dpi=150)
        plt.close()

        # 네트워크 그래프
        plt.figure(figsize=(10, 6))
        plt.plot(times_list, list(self.net_sent_data), 'm-', linewidth=2, label='송신')
        plt.plot(times_list, list(self.net_recv_data), 'c-', linewidth=2, label='수신')
        plt.fill_between(times_list, list(self.net_sent_data), alpha=0.3, color='magenta')
        plt.fill_between(times_list, list(self.net_recv_data), alpha=0.3, color='cyan')
        plt.title('네트워크 트래픽 추이', fontsize=14, fontweight='bold')
        plt.xlabel('시간 (초)', fontsize=12)
        plt.ylabel('속도 (MB/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('network_graph.png', dpi=150)
        plt.close()

        # 코어별 CPU (마지막 값)
        cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        plt.figure(figsize=(10, 6))
        cores = list(range(len(cpu_percent_per_core)))
        bars = plt.bar(cores, cpu_percent_per_core, color='steelblue', alpha=0.7)

        for bar, value in zip(bars, cpu_percent_per_core):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        plt.title('코어별 CPU 사용률 (모니터링 종료 시점)', fontsize=14, fontweight='bold')
        plt.xlabel('CPU 코어', fontsize=12)
        plt.ylabel('사용률 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.xticks(cores, [f'Core {i}' for i in cores])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('cores_graph.png', dpi=150)
        plt.close()


def main():
    """메인 함수"""
    print("=" * 60)
    print("실시간 시스템 리소스 모니터링 시스템")
    print("=" * 60)
    print()

    # 모니터 생성 (5분 = 300초)
    monitor = SystemMonitor(duration=300, update_interval=1)

    # 모니터링 시작
    stats = monitor.start()

    # PDF 리포트 생성
    monitor.generate_pdf_report('system_monitoring_report.pdf')

    print()
    print("=" * 60)
    print("모니터링 완료!")
    print("=" * 60)
    print(f"평균 CPU 사용률: {stats['cpu']['avg']:.2f}%")
    print(f"평균 메모리 사용률: {stats['memory']['avg']:.2f}%")
    print(f"총 네트워크 송신: {stats['net_sent']['total']:.2f} MB")
    print(f"총 네트워크 수신: {stats['net_recv']['total']:.2f} MB")
    print()
    print("PDF 리포트: system_monitoring_report.pdf")
    print("=" * 60)


if __name__ == '__main__':
    main()
