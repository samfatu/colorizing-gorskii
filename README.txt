  The program was developed to revive the photographs taken by the Russian photographer Sergei Mikhailovich Prokudin-Gorskii with an innovative technique
between 1909-1915 in the digital environment today.
  The program takes as input a black and white photograph in which 3 color channels are arranged in a 1:3 ratio. This photo represents the photos
Gorskii took with different filter glasses.
  The program aligns this photo through certain algorithms (SSD), applies filters according to the given arguments, and writes the resulting photo
to a file as a result.

  At the command line to run the program:
     python main.py [args]

   Running the program with arguments is optional. The valid command line argument list is as follows:
     # "-l": This command indicates that you want the channels included in the calculation of the Sum of squared differences (SSD) used during the alignment of the channels of the given photo to be filtered with the Laplacian filter.
     # "-g": This command indicates whether gamma correction is desired after the alignment process is finished.
     # "-s": This command is used to give a sharpening effect to the photo after the alignment process is finished.
     # "-h": This command is used to make histogram equalization to the photo after the alignment process is finished.

   Example program run command:
     python main.py -l -s

   * The order in which the arguments are presented does not matter.

  In order for the program to work, the necessary input photo files must be in the "../html/images/raw_data/" folder.

   While the program is running, it produces a table showing the state of a photo at every stage for each photo it processes. This table consists of 3 photos.
     The first photo shows the divided and colored version of each channel of the photo given as input.
     The second photo shows the color channels aligned with each other.
     The third photo shows the filtered photo with the filter options given before the program runs.
     In this way, you can more easily see the effects of the processes on the photo.
    At the same time, while the plot is displayed at each step, an informative text is written on the command line indicating how many pixels the red and green channels of each photograph are shifted.

  Finally, the final version of each processed photo is written in the "output/" folder with its own name and the prefixes of the applied processes added to the file name.

  Additional information about implementation of the algorithms given in the HTML report in "html/index.html".


  Mustafa Can Genç - 21827426