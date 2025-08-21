# HOPLS

## Overview

HOPLS (Higher-Order Partial Least Squares) extends PLS to model relationships between multi-way (tensor) data and targets. It is useful for extracting compact latent structures from high-dimensional, multi-dimensional datasets where classic matrix-based methods fall short.

## Algorithms

- PLS: Learns latent components that maximize covariance between predictors X and responses Y; effective with multicollinearity and high-dimensional settings.
- HOPLS: Generalizes PLS to tensors (e.g., time × space × variables), preserving multi-way structure for better interpretability and performance on higher-order data.
