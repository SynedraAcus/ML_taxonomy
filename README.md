### Morphology-based classifiers

Testing various classifiers to see whether I can get the quasi-cryptic
species to usefully separate from morphology alone. In all cases (each in its
own subdirectory) the species are known to be distinct from some sort of
evidence (eg molecular phylogeny or interbreeding), but lack a reliable
diagnosis.

`./fragilaria` is our own dataset of *Fragilaria radians*, *Ulnaria acus*, and
*U. ulna*, mostly from lake Baikal. Approx. 30 cells were measured per (clonal)
strain, and for each strain rbcL sequencing was used to determine which species
it is. There are also some hybrids and non-Baikalian strains.

`./aceria` is a dataset of plant parasite mites from
[Skoracka et al. 2014](https://academic.oup.com/biolinnean/article/111/2/421/2415954),
provided by Dr Anna Skoracka. It includes dry bulb mites *Aceria tulipae*
(`DBM`) and two lineages of wheat curl mites *A. tosicella* (`MT-1` and `MT-2`).
The latter two are cryptic (similarly to *U. ulna* vs *U. acus*/*F. radians*
complex from the previous example) 