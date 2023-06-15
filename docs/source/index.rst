.. VSDI Pipeline documentation master file, created by
   sphinx-quickstart on Mon Jun 12 21:54:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VSDI Pipeline
=========================================

Package for handling voltage sensitive dye imaging data

.. graphviz::

   digraph architecture {
       rankdir=LR;
       node [shape=box, style=filled, color=lightblue];

       "VSDI data" [shape=ellipse, color=lightgrey];
       "VSDI data" -> "io";
       "io" -> "session";
       "io" -> "utils";
       "io" -> "dimensionality\nreduction";
       "session" -> "utils";
       "session" -> "dimensionality\nreduction";
       "dimensionality\nreduction" -> "visualization";
       "utils" -> "visualization";
   }

.. toctree::
   :maxdepth: 1
   :caption: Code Documentation

   vsdi.io
   vsdi.linear_dim_red
   vsdi.session
   vsdi.utils
   vsdi.VAE
   vsdi.visualization
