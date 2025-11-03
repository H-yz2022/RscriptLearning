# options("install.packages.compile.from.source" = "never")
rm(list = ls(all.names = TRUE)) # Clear memory 

###- {r}
      is.installed <- function(mypkg) is.element(mypkg, installed.packages()[,1]) 
###- 

###- {r}
if (!is.installed('tm')) { 
    install.packages("tm", dependencies = TRUE, INSTALL_opts = '--no-lock')
    # remove.packages("tm")
    # install.packages("tm")
  }  
library(tm)


# function sets working directory to the file location
setwd_thisdir <- function () {
        this.dir <- dirname(rstudioapi::getSourceEditorContext()$path)
        setwd(this.dir)
}      
      

#   function gets the number of rows in a file
get_review_count_from_file <- function(documents.filename){
  dat = readLines(documents.filename)
  return(length(dat))
}


#   function creates df matrix from a text file where    
#   each line ('\n' separated) represents  one document
get_df_from_file <- function(documents.filename)
  
{
  documents.lines = readLines(fi <- file(documents.filename))
  close(fi)
  return(get_df_from_lines(documents.lines))
}


#   function creates df matrix from an array of text lines
#   each text line represents  one document
get_df_from_lines <- function(documents.lines)
  
{
  
  # Settings for text parsing (tm)
  control <- list(stopwords = TRUE,
                  removePunctuation = TRUE,
                  removeNumbers = TRUE,
                  minDocFreq = 2)
  
  # Create corpus from text lines data  
  
  documents.corpus <- Corpus(VectorSource(documents.lines))
  documents.tdm <- TermDocumentMatrix(documents.corpus, control)
  
  # Create a data frame that provides the feature set from document data
  # Allows for multiple term occurrences per document (i.e., > 1)
  documents.matrix <- as.matrix(documents.tdm)
  
  # BREAKPOINT #1
  #View(documents.matrix[,1:45])
  
  # Term occurrence across all documents 
  documents.counts <- rowSums(documents.matrix)
  

  documents.df <- data.frame(cbind(names(documents.counts),
                                          as.numeric(documents.counts)),
                                          stringsAsFactors = FALSE)
  
  names(documents.df) <- c("term", "frequency")
  
  documents.df$frequency <- as.numeric(documents.df$frequency)
  
  #  occurrence - Share (%) of reviews (documents) that contain the term 
  #    - the same as the probability of finding the term in a randomly selected document from the collection
  #    - used as a conditional probability P(t|c) of the term given class in the binirized NB classifier
  
  documents.occurrence <- sapply(1:nrow(documents.matrix),
                  function(i)
                     {
                        length(which(documents.matrix[i, ] > 0)) / ncol(documents.matrix)
                      })
  

  # Add term occurrence rate
  documents.df <- transform(documents.df, occurrence = documents.occurrence)
  
  ##  documents.df contains 3 columns: 
  ##
  ##  term      -  a word (e.g., "place")
  ##  frequency -  the total number of times the word appears across all reviews 
  ##               (also counting in multiple occurrences per same review/document)
  ##  occurrence - share of reviews that contain the word at least ones 
  
  
  return(documents.df)
  
}

# The function returns the value proportional to the posterior probability that the given review  
# belongs to the class defined by training.df

classify_document <- function(document, training.df, prior, p_t_c_missing)
{
  
  # get word level composition and frequency of the document
  document.df <- get_df_from_lines(document)
  
  # get word vector
  document_words <- as.matrix(document.df$term)
  
  # get the total number of words
  document_num_words <- sum(document.df$frequency)
  
  
  # Find intersections of words in the document and training data
  matching_words <- intersect(document_words, training.df$term)
  
  
  
  # If no words in the document match the words in the training data - return the product of p_t_c_missing probabilities for each term
  if(length(matching_words) < 1)
  {
    
    posterior = prior * p_t_c_missing ^ (sum(document.df$frequency))
    
    #posterior_log = log(prior) +  length(document_words) * log(p_t_c_missing)
    
    return(posterior)
  }
  # otherwise, multiply probabilities of matched words
  else
  {
    
    
    match.probs     <- training.df$p_t_c[match(matching_words, training.df$term)]
    match.occurance <- document.df$frequency[match(matching_words, document.df$term)]
    
    occurance_num_words <- sum(match.occurance)
    missing_p_t_c_count = sum(document.df$frequency) - sum(match.occurance)
    
    # BREAKPOINT #5
    # YOUR CODE IS HERE
    # Replace runif() with your own calculations of posterior distribution
    posterior = runif(1, min = 0, max = 1)
    
    
    
    
    
    
    return(posterior)
    
  }
}


###############################################################################
#
#  MAIN
#
###############################################################################

main <- function () {

 setwd_thisdir()
  
  reviews_positive.file <- "reviews_positive.txt"
  reviews_negative.file <- "reviews_negative.txt"
  reviews_test.file     <- "reviews_test.txt"
  

 nReviewCountPositive = get_review_count_from_file(reviews_positive.file)
 nReviewCountNegative = get_review_count_from_file(reviews_negative.file)
 

 reviews_positive.df <- get_df_from_file(reviews_positive.file)
 reviews_negative.df <- get_df_from_file(reviews_negative.file)

##  xxx.df has 3 columns: 
##
##  term      -  a word (e.g., "place")
##  frequency -  the total number of times the word appears across all reviews 
##               (summing up  multiple occurrences in the same review)
##  occurrence - share of reviews that contain the word at least ones (not used in MN-NB version of the classifier)

nWordsCommon   = length(intersect(reviews_positive.df$term, reviews_negative.df$term))
nWordsPositive = length(reviews_positive.df$term)
nWordsNegative = length(reviews_negative.df$term)

nWords = nWordsPositive + nWordsNegative - nWordsCommon

# BREAKPOINT #2
#View(reviews_positive.df)
#View(reviews_negative.df)

# Creating P(c) (i.e. P("positive") and P("negative"))) 
# as the number of positive reviews to the total number of reviews

P_c_positive = nReviewCountPositive / (nReviewCountPositive + nReviewCountNegative)
P_c_negative = nReviewCountNegative / (nReviewCountPositive + nReviewCountNegative)

reviews_positive.df = transform(reviews_positive.df, 
                                p_t_c = reviews_positive.df$frequency / sum(reviews_positive.df$frequency))

reviews_negative.df = transform(reviews_negative.df, 
                                p_t_c = reviews_negative.df$frequency / sum(reviews_negative.df$frequency))
                                
# BREAKPOINT #3
#View(reviews_positive.df)
#View(reviews_negative.df)


# Creating P(t|c) (i.e. P("term" | "positive") and P("term" | "negative"))
reviews_positive.df = transform(reviews_positive.df, 
                           p_t_c = (reviews_positive.df$frequency + 1) /
                                   (sum(reviews_positive.df$frequency) + nWords))

p_t_c_positive_missing = 1 / (sum(reviews_positive.df$frequency) + nWords)

reviews_negative.df = transform(reviews_negative.df, 
                                p_t_c = (reviews_negative.df$frequency + 1) /
                                  (sum(reviews_negative.df$frequency) + nWords))

p_t_c_negative_missing = 1 / (sum(reviews_negative.df$frequency) + nWords)

# BREAKPOINT #4


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # Run classifier
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Perform classification for a set of reviews

# Load all reviews and class information
Reviews_to_classify <- readLines(fi <- file(reviews_test.file))
close(fi)


# Step 2: Loop over each line and perform class predictions

iIdx = 1
for (sReview in Reviews_to_classify)
{
  prob_class_positive   <-  classify_document(sReview, training.df = reviews_positive.df, 
                                            prior = P_c_positive, p_t_c_missing = p_t_c_positive_missing)


  prob_class_negative   <-  classify_document(sReview, training.df = reviews_negative.df,
                                            prior = P_c_negative, p_t_c_missing = p_t_c_negative_missing)

  if (prob_class_positive > prob_class_negative)
      cat(iIdx, "\tPOSITIVE \t", prob_class_positive, "\t", sReview, "\n")   
   else 
      cat(iIdx, "\tNEGATIVE \t", prob_class_negative, "\t", sReview, "\n")     
  
  iIdx = iIdx + 1
}

}

main()
