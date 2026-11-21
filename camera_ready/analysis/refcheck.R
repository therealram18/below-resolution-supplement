# Run: _refenv/r/bin/R --vanilla -q -f refcheck.R   (from camera_ready/)
.libPaths(c(normalizePath("_refenv/rlib"), .libPaths()))
suppressMessages({library(clubSandwich); library(dfadjust)})
# dfadjustSE 1.1.0 fails for intercept-only models: `Q[ord, ]` drops the matrix to a vector when K = 1.
# Use an in-memory copy with `drop = FALSE` added (nothing else changed; installed package untouched).
src <- paste(deparse(dfadjust::dfadjustSE), collapse = "\n")
stopifnot(grepl("Q <- Q[ord, ]", src, fixed = TRUE))
dfadjustSE <- eval(parse(text = sub("Q <- Q[ord, ]", "Q <- Q[ord, , drop = FALSE]", src, fixed = TRUE)), envir = asNamespace("dfadjust"))
dat <- read.csv("_refenv/pairs_mase_rank.csv", colClasses = c(pair = "character"))
res <- do.call(rbind, lapply(split(dat, factor(dat$pair, levels = unique(dat$pair))), function(g) {
  fit <- lm(d ~ 1, data = g)
  ct <- coef_test(fit, vcov = "CR2", cluster = g$dataset, test = "Satterthwaite")
  cl <- factor(g$dataset)
  a <- dfadjustSE(fit, clustervar = cl, IK = TRUE)
  a0 <- dfadjustSE(fit, clustervar = cl, IK = TRUE, rho0 = TRUE)
  se <- a$coefficients[1, "HC2 se"]; est <- a$coefficients[1, "Estimate"]
  se0 <- a0$coefficients[1, "HC2 se"]
  data.frame(pair = g$pair[1], theta = est, club_se = ct$SE, club_df = ct$df_Satt, club_p = ct$p_Satt,
             dfa_se = se, dfa_df = a$coefficients[1, "df"], dfa_p = 2 * pt(-abs(est / se), a$coefficients[1, "df"]),
             dfa_rho_ik = a$rho, dfa_sig = a$sig, dfa_adjse = a$coefficients[1, "Adj. se"],
             dfa0_df = a0$coefficients[1, "df"], dfa0_p = 2 * pt(-abs(est / se0), a0$coefficients[1, "df"]),
             dfa0_rho_ik = a0$rho, dfa0_sig = a0$sig,
             dfa_noik_df = dfadjustSE(fit, clustervar = cl, IK = FALSE)$coefficients[1, "df"])
}))
write.csv(res, "_refenv/ref_R.csv", row.names = FALSE)
cat(R.version.string, "\n"); print(summary(res[, c("club_df", "dfa_df", "dfa_rho_ik")]))
# fwildclusterboot 0.14 installs but boottest() errors on an intercept-only lm (matrix drops to a vector when K = 1,
# for both fnw11 and 11); not used. Versions:
for (p in c("clubSandwich", "dfadjust", "sandwich", "lmtest", "fwildclusterboot")) cat(p, as.character(packageVersion(p)), "\n")
