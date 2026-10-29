"""Shared chart typography and paper; never changes values or axis scales."""


def style_figure(figure, dark=False):
    paper, plot, ink, muted, edge = (
        ('#1a2133', '#20283b', '#edf0fa', '#bac4da', '#39435b') if dark else
        ('#ffffff', '#faf8ff', '#28223d', '#635a78', '#e0d8ef'))
    figure.set_facecolor(paper)
    for axis in figure.axes:
        axis.set_facecolor(plot)
        axis.title.set(fontsize=14, fontweight='bold', color=ink)
        axis.title.set_position((.5, 1.025))
        axis.xaxis.label.set(fontsize=10, color=muted)
        axis.yaxis.label.set(fontsize=10, color=muted)
        axis.tick_params(colors=muted, labelsize=9, length=0, pad=8)
        axis.xaxis.get_offset_text().set_color(muted)
        axis.yaxis.get_offset_text().set_color(muted)
        for name, spine in axis.spines.items():
            spine.set_color(edge)
            spine.set_visible(name in {'bottom', 'left'})
        for line in [*axis.get_xgridlines(), *axis.get_ygridlines()]:
            line.set_color(edge)
            line.set_alpha(.65)
            line.set_linewidth(.7)
        legend = axis.get_legend()
        if legend:
            legend.get_frame().set_facecolor(paper)
            legend.get_frame().set_edgecolor(edge)
            legend.get_frame().set_alpha(.95)
            for text in legend.get_texts():
                text.set(color=ink, fontsize=9)
    if figure._suptitle:
        figure._suptitle.set(color=ink, fontsize=16, fontweight='bold')
