library(matrixStats)
library(ggplot2)

plot_known_viol = function(
        x, px, py, line=NULL, point=NULL, range1=NULL, range2=NULL, g=NULL,
        colors='gray', width=NULL) {

    # number of violins
    n = length(x)

    # template colors
    if (colors == 'gray') {
        colors = list(
            fill    = '#bfbfbf',
            edge    = '#e6e6e6',
            range1 = '#606060',
            range2 = '#909090',
            line  = '#000000',
            point    = '#ffffff'
        )
    } else if (colors == 'green') {
        colors = list(
            fill    = '#b8d997',
            edge    = '#e6f1da',
            range1 = '#80ba45',
            range2 = '#606060',
            line  = '#000000',
            point    = '#ffffff'
        )
    } else if (colors == 'green_strong') {
        colors = list(
            fill    = '#bae4b3',
            edge    = '#e5f5e0',
            range1 = '#41ab5d',
            range2 = '#cc6666',
            line  = '#000000',
            point    = '#ffffff'
        )
    } else if (colors == 'blue') {
        colors = list(
            fill    = '#bdd7e7',
            edge    = '#deebf7',
            range1 = '#4292c6',
            range2 = '#cc6666',
            line  = '#000000',
            point    = '#ffffff'
        )
    }
    # repeat singleton colors
    colors = lapply(colors, function(e) if(length(e)==1) rep(e, n) else e)

    # init plot object
    if (is.null(g))
        g = ggplot()

    # calculate violin scalin
    if (is.null(width))
        width = min(
            1.2 * min(x[2:length(x)] - x[1:length(x)-1]),  # 1.2 * min[dx]
            0.25 * (max(x) - min(x))
        )
    scale = 0.5 * width / rowMaxs(py)

    # calc line edges
    if (!is.null(line)) {
        # find violin edges
        l_inds = apply(line-px, 1, FUN = function(x) {max(which(x >= 0))})
        l_y1 = py[seq(1,n) + (l_inds-1)*n]
        l_y2 = py[seq(1,n) + l_inds*n]
        l_x1 = px[seq(1,n) + (l_inds-1)*n]
        l_x2 = px[seq(1,n) + l_inds*n]
        l_k = (l_y2-l_y1) / (l_x2-l_x1)
        l_y = l_y1 + (line - l_x1) * l_k
    }

    # add violin bodies
    for (i in 1:n) {
        g = g + geom_polygon(
            data = data.frame(
                x = x[i] + scale[i]*c(-py[i,], rev(py[i,])),
                y = c(px[i,], rev(px[i,]))
            ),
            aes(x=x, y=y),
            color = colors$edge[i],
            fill = colors$fill[i]
        )
        # add line segment
        if (!is.null(line)) {
            g = g + geom_segment(
                data = data.frame(
                    x=x[i]-l_y[i]*scale[i], y=line[i],
                    xend=x[i]+l_y[i]*scale[i], yend=line[i]),
                aes(x=x, y=y, xend=xend, yend=yend),
                color = colors$line[i]
            )
        }
    }

    # add range1
    if (!is.null(range1)) {
        g = g + geom_linerange(
            data = data.frame(x=x, ymin=range1[,1], ymax=range1[,2]),
            aes(x=x, ymin=ymin, ymax=ymax),
            color = colors$range1
        )
    }

    # add range2
    if (!is.null(range2)) {
        g = g + geom_linerange(
            data = data.frame(x=x, ymin=range2[,1], ymax=range2[,2]),
            aes(x=x, ymin=ymin, ymax=ymax),
            color = colors$range2,
            size = 2
        )
    }

    # add point
    if (!is.null(point)) {
        g = g + geom_point(
            data = data.frame(x=x, y=point),
            aes(x=x, y=y),
            color = colors$point
        )
    }

    # return
    g
}
