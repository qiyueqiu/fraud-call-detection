"""
图表保存工具:同时输出矢量图(PDF, 供论文使用)与位图(PNG, 供 README/预览使用)。
论文 main.tex 引用 PDF 矢量图以保证印刷质量。
"""
import os


def savefig_dual(path_png, **kwargs):
    """传入 .png 路径,自动同时保存同名 .pdf(矢量)与 .png(位图)。"""
    import matplotlib.pyplot as plt
    kwargs.setdefault("bbox_inches", "tight")
    base, _ = os.path.splitext(path_png)
    # 矢量图(论文用)
    plt.savefig(base + ".pdf", **{k: v for k, v in kwargs.items() if k != "dpi"})
    # 位图(预览用)
    kwargs.setdefault("dpi", 150)
    plt.savefig(base + ".png", **kwargs)
