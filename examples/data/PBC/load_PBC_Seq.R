# Title     : PBC_Seq
# Objective : load
# Created by: Van Tuan NGUYEN
# Created on: 24/11/2023

defaultW <- getOption("warn")
options(warn = -1)
library("JMbayes")

load <- function() {

    data(pbc2, package = "JMbayes")
    data <- pbc2
    data$times <- data$year
    data$tte <- data$years
    data$label <- data$status2

    return(data)
}