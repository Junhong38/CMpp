import plotly.graph_objects as go
import os
import torch
import numpy as np

class MatchVisualizer:
    def create_visualization(self, source_points, target_points, correspondences, src_ori, trg_ori, greenline, rrmse, trmse, title="Correspondences Visualization"):
        """
        소스/타겟 포인트 클라우드 및 correspondence 시각화

        Args:
            source_points (np.ndarray): (N, 3) source point cloud
            target_points (np.ndarray): (M, 3) target point cloud
            correspondences (np.ndarray): (K, 2) 각 행이 (source_idx, target_idx) 쌍

        Returns:
            go.Figure: 시각화된 Plotly Figure 객체
        """
        fig = go.Figure()

        # Source points (orange)
        fig.add_trace(go.Scatter3d(
            x=source_points[:, 0], y=source_points[:, 1], z=source_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='orange'),
            name='Source'
        ))

        # Target points (green)
        fig.add_trace(go.Scatter3d(
            x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='green'),
            name='Target'
        ))

        # Correspondence lines (blue: true matches, red: false matches)
        vis_in, vis_out = 0, 0
        if correspondences.size(0) == len(target_points):
            for i, (src_idx, tgt_idx) in enumerate(correspondences):
                p1 = source_points[i]
                p2 = target_points[i]
                is_match = (torch.tensor([src_idx, tgt_idx], device=correspondences.device) == correspondences[greenline]).all(dim=1).any()
                if len(greenline) > 0 and is_match:
                    # breakpoint()
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        showlegend=False
                    ))
                    vis_in += 1
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ))
                    vis_out += 1

            arrow_scale = 0.005  # 길이 조절
            # Source 방향 시각화
            for i in range(len(source_points)):
                src_pos = np.array(source_points[i])
                for j in range(3):  # x, y, z 방향
                    src_vec = np.array(src_ori[i][j])
                    arrow_end = src_pos + arrow_scale * src_vec

                    fig.add_trace(go.Scatter3d(
                        x=[src_pos[0], arrow_end[0]],
                        y=[src_pos[1], arrow_end[1]],
                        z=[src_pos[2], arrow_end[2]],
                        mode='lines',
                        line=dict(color='green', width=3),
                        showlegend=False
                    ))
            # Target 방향 시각화
            for i in range(len(target_points)):
                tgt_pos = np.array(target_points[i])
                for j in range(3):  # x, y, z 방향
                    tgt_vec = np.array(trg_ori[i][j])
                    arrow_end = tgt_pos + arrow_scale * tgt_vec

                    fig.add_trace(go.Scatter3d(
                        x=[tgt_pos[0], arrow_end[0]],
                        y=[tgt_pos[1], arrow_end[1]],
                        z=[tgt_pos[2], arrow_end[2]],
                        mode='lines',
                        line=dict(color='purple', width=3),
                        showlegend=False
                    ))

        else:
            for src_idx, tgt_idx in correspondences:
                if src_idx < len(source_points) and tgt_idx < len(target_points):
                    p1 = source_points[src_idx]
                    p2 = target_points[tgt_idx]
                    is_match = (torch.tensor([src_idx, tgt_idx], device=correspondences.device) == correspondences[greenline]).all(dim=1).any()
                    if len(greenline) > 0 and is_match:
                        # breakpoint()
                        fig.add_trace(go.Scatter3d(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                            mode='lines',
                            line=dict(color='blue', width=2),
                            showlegend=False
                        ))
                        vis_in += 1
                    else:
                        fig.add_trace(go.Scatter3d(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                            mode='lines',
                            line=dict(color='red', width=2),
                            showlegend=False
                        ))
                        vis_out += 1
        
        if type(greenline) == list:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0.01, color='rgba(0,0,0,0)'),
                name=f"total True match: 0/{len(correspondences)}"
            ))
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0.01, color='rgba(0,0,0,0)'),
                name=f"total False match: {len(correspondences)}/{len(correspondences)}"
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0.01, color='rgba(0,0,0,0)'),
                name=f"total True match: {torch.sum(greenline).item()}/{len(correspondences)}"
            ))
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=0.01, color='rgba(0,0,0,0)'),
                name=f"total False match: {greenline.size(0) - torch.sum(greenline).item()}/{len(correspondences)}"
            ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=0.01, color='rgba(0,0,0,0)'),
            name=f"rotation rsme: {rrmse}"
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=0.01, color='rgba(0,0,0,0)'),
            name=f"translation rsme: {trmse}"
        ))


        # 레이아웃 설정
        fig.update_layout(
            title=title,
            scene=dict(aspectmode='data'),
            width=1000,
            height=800,
            margin=dict(l=10, r=10, t=50, b=10)
        )

        return fig
        
    def save_fragments_visualization(self, src_pts, tar_pts, corr, src_ori, trg_ori, vis_num_pair, greenline, registration, initial_match, vis_inlier, rrmse, trmse, output_dir="visualize_correspondence/output"):
        """
        파편 시각화 파일 저장 (각 도형별로 별도 디렉토리에 HTML 파일 저장)
        
        Args:
            fragments: 파편 데이터 리스트
            output_dir: 출력 디렉토리
        
        Returns:
            tuple: (저장된 HTML 파일 경로 리스트, HTML 파일 총 용량)
        """
        # 기본 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        html_files = []
        html_total_size = 0  # HTML 파일 총 용량 (바이트)
        src_corr_ori = src_ori[corr[:,0]] if src_ori != None else None
        trg_corr_ori = trg_ori[corr[:,1]] if trg_ori != None else None
        # 전체 파편 시각화는 최상위 디렉토리에 저장
        all_fig = self.create_visualization(src_pts, tar_pts, corr, src_corr_ori, trg_corr_ori, greenline, rrmse, trmse, "Correspondences Visualization")
        all_fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=80, b=10),
            title_x=0.5,
            font=dict(size=16)
        )
        
        all_fragments_path = os.path.join(output_dir, f"{registration}_{initial_match}_inlier:{vis_inlier}_{vis_num_pair}.html")
        all_fig.write_html(
            all_fragments_path,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
            include_mathjax=False,
            config={'responsive': True, 'displayModeBar': True}
        )
        html_files.append(all_fragments_path)
        
        # 파일 크기 추적
        if os.path.exists(all_fragments_path):
            html_total_size += os.path.getsize(all_fragments_path)
        
        return html_files, html_total_size