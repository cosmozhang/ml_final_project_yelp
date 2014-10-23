brcdata=read.csv('./biz_review_counts_AZ.csv', header = FALSE, sep=":")
Fn <- ecdf(brcdata$V2)
plot(Fn, verticals = TRUE, do.points = FALSE)
length(which(brcdata$V2>200))
length(which(brcdata$V2>200))/length(which(brcdata$V2>0))
brcdata$V1[(which(brcdata$V2>200))]