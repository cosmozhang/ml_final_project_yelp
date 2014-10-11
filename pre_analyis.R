urcdata=read.csv('./user_review_counts.csv', header = FALSE, sep=":")

png(file="uzcounts.png")
par(mfrow = c(1,1))
hist(urcdata$V2, breaks=100, freq=FALSE, main="user review counts distribution", col="purple")
curve(dnorm(x, mean=mean(urcdata$V2), sd=sd(urcdata$V2)), add=TRUE, 
      col="red", lwd=2)

dev.off()

brcdata=read.csv('./biz_review_counts.csv', header = FALSE, sep=":")

png(file="bizcounts.png")
par(mfrow = c(1,1))
hist(brcdata$V2, breaks=100, freq=FALSE, main="biz review counts distribution", col="green")
curve(dnorm(x, mean=mean(brcdata$V2), sd=sd(brcdata$V2)), add=TRUE, 
      col="red", lwd=2)

dev.off()