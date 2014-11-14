brcdata=read.csv('./biz_review_counts_AZ.csv', header = FALSE, sep=":")
Fn <- ecdf(brcdata$V2)
plot(Fn, verticals = TRUE, do.points = FALSE)
length(which(brcdata$V2>200))
length(which(brcdata$V2>200))/length(which(brcdata$V2>0))
brcdata$V1[(which(brcdata$V2>200))]

png(file="uzcounts.png")
par(mfrow = c(1,1))
hist(urcdata$V2, breaks=10000, freq=TRUE, main="", col="purple", xlim=c(0, 50), xlab = 'User review counts')
#curve(dnorm(x, mean=mean(urcdata$V2), sd=sd(urcdata$V2)), add=TRUE, col="red", lwd=2)

dev.off()

brcdata=read.csv('./biz_review_counts.csv', header = FALSE, sep=":")

png(file="bizcounts.png")
par(mfrow = c(1,1))
hist(brcdata$V2, breaks=1000, freq=TRUE, main="", col="green", xlim=c(0, 50), xlab = 'Biz review counts')
#curve(dnorm(x, mean=mean(brcdata$V2), sd=sd(brcdata$V2)), add=TRUE, col="red", lwd=2)

dev.off()
