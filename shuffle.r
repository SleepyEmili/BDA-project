setwd("C:\\Users\\User\\Desktop\\Classwork\\COMP4434\\BDA-project")
data = read.csv("virusshare.csv",header = T)
shuffled_data = data[sample(1:nrow(data)), ]
write.csv(shuffled_data,"virusshare2.csv", row.names = F)
