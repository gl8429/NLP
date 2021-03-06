ATIS <- c(10.591, 11.636, 7.235)
WSJ <- c(88.890, 86.660, 46.514)
BROWN <- c(113.360, 110.783, 61.469)
g_range <- range(0, ATIS, WSJ, BROWN)
plot_colors <- c("blue", "red", "forestgreen")
plot(ATIS, type = "o", col = plot_colors[1], ylim = g_range, axes = FALSE, ann = FALSE)
axis(1, at = 1:3, lab = c("Forward", "Backward", "Bidirectional"))
axis(2, las=1, at=20*0:g_range[2])
box()
lines(WSJ, type="o", pch=22, lty=2, col=plot_colors[2])
lines(BROWN, type="o", pch=22, lty=2, col=plot_colors[3])
title(main="Word Preplexity of Training Data", col.main="red", font.main=4)
title(xlab= "Bigram Model", col.lab=rgb(0,0.5,0))
title(ylab= "Word Preplexity", col.lab=rgb(0,0.5,0))
legend(1, g_range[2]*0.5, c("ATIS", "WSJ", "BROWN"), cex=0.8, col=plot_colors, pch=21:23, lty=1:3);