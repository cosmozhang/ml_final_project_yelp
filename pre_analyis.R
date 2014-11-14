urcdata=read.csv('./user_review_counts.csv', header = FALSE, sep=":")

png(file="uzcounts.png")
par(mfrow = c(1,1))
hist(urcdata$V2, breaks=1000, freq=TRUE, main="user review counts distribution", col="purple", xlim=c(0, 100))
curve(dnorm(x, mean=mean(urcdata$V2), sd=sd(urcdata$V2)), add=TRUE, 
      col="red", lwd=2)

dev.off()

brcdata=read.csv('./biz_review_counts.csv', header = FALSE, sep=":")

png(file="bizcounts.png")
par(mfrow = c(1,1))
hist(brcdata$V2, breaks=1000, freq=TRUE, main="biz review counts distribution", col="green", xlim=c(0, 100))
curve(dnorm(x, mean=mean(brcdata$V2), sd=sd(brcdata$V2)), add=TRUE, 
      col="red", lwd=2)

dev.off()

cityrv_data=read.csv('avg_review_city.csv', header=F, sep=',')
print(summary(cityrv_data))
avgc = matrix(cityrv_data$V4,ncol=length(cityrv_data$V4),byrow=TRUE)
colnames(avgc) = cityrv_data$V1
city_v =  as.table(avgc)
png(file="cityrv.png")
barplot(city_v, main = "city", col =1:1000, las=2)
dev.off()

