plot_label <- function(text) {
  # R's PDF device maps ASCII '-' to minus; character 173 selects the hyphen glyph.
  if (is.character(x = text) && names(x = dev.cur()) == "pdf") {
    return(gsub(pattern = "-", replacement = "\u00ad", x = text, fixed = TRUE))
  }
  text
}
