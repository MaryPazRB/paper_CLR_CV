# Data Analysis for Coffee Leaf Rust Severity
# Author: [User's Name/Project Group]
# Description: Comparative analysis of segmentation methods (ImageJ, pliman, DeepLabV3+, SAM_CLR, SAM3).

# ==============================================================================
# 1. Setup and Libraries
# ==============================================================================
library(readxl)
library(dplyr)
library(tidyverse)
library(purrr)
library(epiR)
library(cowplot)

# Set global plotting theme
theme_set(cowplot::theme_half_open(font_size = 12))

# ==============================================================================
# 2. Data Loading
# ==============================================================================
# Function to load and clean CSVs ensuring numeric types where needed
load_csv <- function(path) {
  read.csv(path, stringsAsFactors = FALSE)
}

# Load Gold Standard (Severity Index)
# Removed R1OLD, R2OLD, R1NEW, R2NEW. Renamed SAM3_2 as SAM3.
severity_final <- load_csv("severity_final.csv") %>%
  select(-R1OLD, -R2OLD, -R1NEW, -R2NEW, -SAM3) %>%
  rename(SAM3 = SAM3_2) %>%
  rename(SAM_CLR = SAM100) %>%
  mutate(across(-image, as.numeric))

# Load Method Results
deeplab <- load_csv("deeplab.csv")
imagej <- load_csv("ImageJ.csv")
pliman <- load_csv("pliman.csv")
SAM_CLR <- load_csv("SAM2.csv")
SAM3 <- load_csv("SAM3.csv")

# ==============================================================================
# 3. Helper Functions
# ==============================================================================

#' Extract CCC results from epiR object
extract_epi_ccc <- function(res) {
  data.frame(
    CCC        = res$rho.c$est,
    CCC.lwr95  = res$rho.c$lower,
    CCC.upr95  = res$rho.c$upper,
    r          = res$rho,
    Cb         = res$C.b
  )
}


#' Compute R-squared and RSE
compute_r2_rse <- function(gs, method) {
  fit <- lm(method ~ gs)
  data.frame(
    R_squared = summary(fit)$r.squared,
    RSE       = summary(fit)$sigma
  )
}

#' Summarise segmentation metrics (IoU, Dice, etc.)
summarise_metrics <- function(df, method_name, fun = median) {
  df %>%
    mutate(across(c(iou, dice, precision, recall, f1), ~ as.numeric(gsub(",", "", .x)))) %>%
    summarise(
      Method    = method_name,
      IoU       = fun(iou, na.rm = TRUE),
      Dice      = fun(dice, na.rm = TRUE),
      Precision = fun(precision, na.rm = TRUE),
      Recall    = fun(recall, na.rm = TRUE),
      F1        = fun(f1, na.rm = TRUE)
    )
}

#' Plot segmentation metrics vs gold standard severity
plot_method_vs_severity <- function(df_metrics, method_name, gs_data) {
  # Prepare data for plotting
  df_long <- df_metrics %>%
    select(image_id, iou, dice, precision, recall, f1) %>%
    left_join(select(gs_data, image_id, GS), by = "image_id") %>%
    mutate(
      GS = as.numeric(GS),
      across(c(iou, dice, precision, recall, f1), ~ as.numeric(gsub(",", "", .x)))
    ) %>%
    pivot_longer(
      cols = c(iou, dice, precision, recall, f1),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = case_when(
        metric == "iou" ~ "IoU",
        TRUE ~ toupper(metric)
      )
    ) %>%
    filter(!is.na(GS), !is.na(value))

  # Create plot
  ggplot(df_long, aes(x = GS, y = value)) +
    geom_point(alpha = 0.6, size = 1.5, aes(color = metric)) +
    geom_smooth(method = "loess", span = 0.75, se = FALSE, linewidth = 1, color = "black") +
    facet_wrap(~metric, ncol = 3) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_viridis_d() +
    labs(
      title = paste("Segmentation performance vs severity —", ifelse(method_name == "pliman", "pliman", toupper(method_name))),
      x = "Gold standard severity",
      y = "Metric value"
    ) +
    theme(
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    )
}

# ==============================================================================
# 4. Analysis: Agreement (CCC)
# ==============================================================================
gold_standard <- "GS"
# Ordered as requested: IMAGEJ, PLIMAN, DEEPLABV3, SAM_CLR AND SAM3
methods_cols <- c("ImageJ", "Pliman", "DeeplabV3", "SAM_CLR", "SAM3")

# Calculate CCC and Regression stats for each method
final_results <- map_dfr(methods_cols, function(m) {
  res_ccc <- epi.ccc(
    x = severity_final[[gold_standard]],
    y = severity_final[[m]],
    ci = "z-transform",
    conf.level = 0.95
  )

  cbind(
    Method = m,
    extract_epi_ccc(res_ccc),
    compute_r2_rse(severity_final[[gold_standard]], severity_final[[m]])
  )
})

# Plot Agreement
plot_ccc <- final_results %>%
  arrange(CCC) %>%
  ggplot(aes(x = CCC, y = reorder(Method, CCC))) +
  geom_point(size = 3) +
  geom_errorbar(aes(xmin = CCC.lwr95, xmax = CCC.upr95), height = 0.2, orientation = "y" ) +
  geom_vline(xintercept = 0.90, linetype = "dashed") +
  scale_x_continuous(limits = c(0.3, 1)) +
  scale_color_viridis_d() +
  labs(
    title = "Agreement with Gold Standard (CCC)",
    x = "Lin’s CCC",
    y = "Method"
  )

print(plot_ccc)

ggsave("ccc_agreement.png", plot_ccc, width = 8, height = 6, dpi = 300)

# ==============================================================================
# 5. CCC Scatterplots (All dots + CCC stats like publication figures)
# ==============================================================================

plot_ccc_method <- function(df, method_col, gs_col = "GS", method_name = NULL) {
  
  x <- df[[gs_col]]
  y <- df[[method_col]]
  
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  
  # ---- CCC ----
  res <- epi.ccc(x, y, ci = "z-transform", conf.level = 0.95)
  
  rho_c <- res$rho.c$est
  lwr   <- res$rho.c$lower
  upr   <- res$rho.c$upper
  r     <- res$rho
  Cb    <- res$C.b
  
  # ---- Regression ----
  fit   <- lm(y ~ x)
  slope <- coef(fit)[2]
  mu    <- coef(fit)[1]
  rmse  <- sqrt(mean((y - x)^2))
  bias  <- mean(y - x)
  
  title <- ifelse(is.null(method_name), method_col, toupper(method_name))
  
  subtitle <- sprintf(
    "ρc = %.2f [%.2f–%.2f], Cb = %.2f, r = %.2f\nμ = %.2f, β = %.2f, RMSE = %.2f, Bias = %.2f",
    rho_c, lwr, upr, Cb, r, mu, slope, rmse, bias
  )
  
  ggplot(data.frame(x, y), aes(x, y)) +
    geom_point(color = "#F28E2B",size = 2, alpha = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.8) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Reference Severity (%)",
      y = "Predicted Severity (%)"
    ) +
    coord_equal(xlim = c(0, 60), ylim = c(0, 60)) +
    theme_half_open(font_size = 12)
}

# ---- Create plots for all methods in the requested order ----
plots_ccc <- map(methods_cols, ~ plot_ccc_method(
  severity_final,
  method_col = .x,
  gs_col = gold_standard,
  method_name = .x
))

# ---- Combine into one figure ----
figure_ccc <- plot_grid(plotlist = plots_ccc, ncol = 3)

print(figure_ccc)

ggsave("CCC_scatter_models.png", figure_ccc,
       width = 10, height = 7, dpi = 300)

# ==============================================================================
# 6. Analysis: Metrics Summary
# ==============================================================================
method_names <- c("deeplab", "imagej", "pliman", "SAM_CLR", "SAM3")

# Generate Median and Mean summary tables
metrics_median <- map_dfr(method_names, ~ summarise_metrics(get(.x), .x, fun = median))
metrics_mean <- map_dfr(method_names, ~ summarise_metrics(get(.x), .x, fun = mean))

cat("\n--- Median Metrics ---\n")
print(metrics_median)
cat("\n--- Mean Metrics ---\n")
print(metrics_mean)

# ==============================================================================
# 7. Analysis: Visualization
# ==============================================================================
# Clean Image IDs (remove extensions)
clean_id <- function(df) mutate(df, image_id = sub("\\.[Pp][Nn][Gg]$", "", image))

severity_final <- clean_id(severity_final)
methods_list <- list(
  deeplab = clean_id(deeplab),
  imagej  = clean_id(imagej),
  pliman  = clean_id(pliman),
  SAM_CLR    = clean_id(SAM_CLR),
  SAM3    = clean_id(SAM3)
)

# Plot 1: Performance vs Severity for each method
plots_perf <- imap(methods_list, ~ plot_method_vs_severity(.x, .y, severity_final))

# Display all performance plots
walk(plots_perf, print)

# Plot 2: Density Distribution of Metrics
# Combine all data into one long format dataframe
metrics_long <- imap_dfr(methods_list, ~ .x %>% mutate(Method = .y)) %>%
  select(image_id, Method, iou, dice, precision, recall, f1) %>%
  left_join(select(severity_final, image_id, GS), by = "image_id") %>%
  mutate(across(c(iou, dice, precision, recall, f1), ~ as.numeric(gsub(",", "", .x)))) %>%
  pivot_longer(
    cols = c(iou, dice, precision, recall, f1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    Method = ifelse(Method == "pliman", "pliman", toupper(Method)),
    metric = case_when(
      metric == "iou" ~ "IoU",
      TRUE ~ toupper(metric)
    )
  ) %>%
  filter(!is.na(value))

# Calculate medians for the plot overlay
metrics_medians <- metrics_long %>%
  group_by(Method, metric) %>%
  summarise(grp_median = median(value, na.rm = TRUE), .groups = "drop")

# Create Density Plot
plot_density <- ggplot(metrics_long, aes(x = value, fill = metric)) +
  geom_density(alpha = 0.6, color = NA) +
  # Add dashed vertical line for the median
  geom_vline(
    data = metrics_medians,
    aes(xintercept = grp_median),
    linetype = "dashed",
    linewidth = 0.4
  ) +
  geom_text(
    data = metrics_medians,
    aes(x = 0, y = Inf, label = paste0("tilde(x):~", round(grp_median, 2))),
    parse = TRUE,
    hjust = -0.1,
    vjust = 5,
    size = 3,
    inherit.aes = FALSE
  ) +
  facet_grid(Method ~ metric) +
  scale_fill_viridis_d() +
  # Clean x-axis labels (remove trailing decimal zeros)
  scale_x_continuous(
    breaks = seq(0, 1, 0.25),
    labels = function(x) ifelse(x %in% c(0, 1), as.character(x), as.character(x))
  ) +
  labs(
    x = "Metric value",
    y = "Density",
    fill = "Metric",
    title = "Distribution of segmentation metrics by method"
  ) +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

print(plot_density)

ggsave("models.png", plot_density, bg = "white", width = 8, height = 6, dpi = 300)



# ============================================================
# gt + gtExtras: metrics in columns, with (median + mini-plot)
# Requires: metrics_long with columns Method, metric, value
# ============================================================

library(tidyr)
library(gt)
library(gtExtras)

# --- 0) (Optional) ensure expected order/names of metrics ---
metrics  <- c("IoU","DICE","PRECISION","RECALL","F1")
methods  <- c("pliman", "IMAGEJ", "DEEPLAB", "SAM_CLR", "SAM3")

metrics_long2 <- metrics_long %>%
  filter(!is.na(value)) %>%
  mutate(
    metric = factor(metric, levels = metrics),
    Method = factor(Method, levels = methods)
  )
# --- 1) List-column with values (to become a mini-plot) ---
dist_wide <- metrics_long2 %>%
  group_by(Method, metric) %>%
  summarise(dist = list(as.numeric(value)), .groups = "drop") %>%
  mutate(metric = as.character(metric)) %>%
  pivot_wider(names_from = metric, values_from = dist)

# --- 2) Medians per metric (becomes numeric columns) ---
med_wide <- metrics_long2 %>%
  group_by(Method, metric) %>%
  summarise(med = median(as.numeric(value), na.rm = TRUE), .groups = "drop") %>%
  mutate(metric = as.character(metric)) %>%
  pivot_wider(names_from = metric, values_from = med, names_prefix = "med_")

# --- 3) Combine everything (1 row per Method) ---
tbl_wide <- dist_wide %>%
  left_join(med_wide, by = "Method") %>%
  mutate(Method = factor(Method, levels = methods)) %>%
  arrange(Method)

# --- 4) gt Table: mini-plots + spanners with unique id ---
tab <- tbl_wide %>%
  gt(rowname_col = "Method") %>%
  # format medians
  fmt_number(columns = all_of(paste0("med_", metrics)), decimals = 2) %>%
  # mini-plots (one per metric column with list-column)
  { purrr::reduce(metrics, .init = ., .f = \(gt_tbl, m) {
    gt_tbl %>% gt_plt_dist(column = all_of(m), fill = "steelblue", type = "density")
  })
  } %>%
  # labels
  cols_label(.list = c(
    setNames(rep("Median", length(metrics)), paste0("med_", metrics)),
    setNames(rep("Dist.",  length(metrics)), metrics)
  )) %>%
  # spanners per metric (with unique id to avoid errors)
  { purrr::reduce(metrics, .init = ., .f = \(gt_tbl, m) {
    gt_tbl %>%
      tab_spanner(
        label = m,
        columns = c(paste0("med_", m), m),
        id = paste0("sp_", m)
      )
  })
  } %>%
  # column widths (fine tuning)
  cols_width(
    all_of(paste0("med_", metrics)) ~ px(70),
    all_of(metrics) ~ px(100)
  ) %>%
  cols_align(align = "center") %>%
  # font style, sizes and borders
  tab_options(
    table.font.names = "Arial",
    column_labels.font.size = px(20),
    column_labels.font.weight = "bold",
    table.font.size = px(16),
    data_row.padding = px(15), 
    # Black borders (top and bottom only)
    table.border.top.color = "black",
    table.border.bottom.color = "black",
    column_labels.border.top.color = "black",
    column_labels.border.bottom.color = "black",
    # Remove internal lines
    table_body.hlines.style = "none",
    column_labels.border.bottom.width = px(2), # keep a slightly thicker line below labels
    table_body.border.bottom.color = "black",
    stub.border.style = "none"
  ) %>%
  tab_style(
    style = cell_text(size = px(22), weight = "bold", align = "center"),
    locations = cells_column_spanners()
  )

tab
