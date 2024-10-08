
/********************************************************************
                           G R I D G E N


This network generator generates a grid-like network plus a super
node.  In additional to the arcs connecting the nodes in the grid,
there is an arc from each supply node to the super node and from the
super node to each demand node to guarantee feasiblity.  These arcs
have very high costs and very big capacities.

The idea of this network generator is as follows: First, a grid of n1
* n2 is generated.  For example, 5 * 3.  The nodes are numbered as 1
to 15, and the supernode is numbered as n1*n2+1.  Then arcs between
adjacent nodes are generated.  For these arcs, the user is allowed to
specify either to generate two-way arcs or one-way arcs.  If two-way
arcs are to be generated, two arcs, one in each direction, will be
generated between each adjacent node pairs.  Otherwise, only one arc
will be generated.  If this is the case, the arcs will be generated in
alterntive directions as shown below.

                1 ---> 2 ---> 3 ---> 4 ---> 5
                |      ^      |      ^      |
                |      |      |      |      |
                V      |      V      |      V
                6 <--- 7 <--- 8 <--- 9 <--- 10
                |      ^      |      ^      |
                |      |      |      |      |
                V      |      V      |      V
               11 --->12 --->13 --->14 ---> 15

Then the arcs between the super node and the source/sink nodes are
added as mentioned before.  If the number of arcs still doesn't reach
the requirement, additional arcs will be added by uniformly picking
random node pairs.  There is no checking to prevent multiple arcs
between any pair of nodes.  However, there will be no self-arcs (arcs
that poins back to its tail node) in the network.

The source and sink nodes are selected uniformly in the network, and
the imbalances of each source/sink node are also assigned by uniform
distribution.

The program can either read the input parameters from the standard
input or from a batch file.  To read from the standard input, just
type

        gridgen > outputfile

or

        gridgen < datafile > outputfile

at the UNIX prompt.  In this mode only one network can be generated at
a time and THE OUTPUT WILL BE SENT TO THE STANDARD OUTPUT.  If data
are stored in a batch file, name of the batch file should be given at
the command line as

        gridgen batchfile

In this mode multiple sets of data can be generated at a time.  Simply
stack the sets of data into the same file.  Comments can be added
after each data.  Comment lines can also be added in the file by
putting an '#' at the first column of the line.

Batch files have the format as shown below.  Each input data must
occupy a line and the data itself should be at the very beginning of
each line.  Characters following the leading numbers/characters are
considered as comments.

The format of the batch file is:

# of jobs:        more than one job can be put in an input file
output file name: name of the file to store the generated network
two-way arcs:     enter 1 if links in both direction should be generated,
                  0 otherwise
random number seed: a positive integer
# of nodes:       the number of nodes generated might be slightly
                  different from specified to make the network a grid.
grid width
# of sources
# of sinks
average degree
total flow
distribution of arc costs:
    1 for UNIFORM
      1st parameter for arc cost (float): lower bound
      2nd parameter for arc cost (float): upper bound
    2 for EXPONENTIAL
      parameter for lambda
distribution of arc capacities:
    1 for UNIFORM
      1st parameter for arc capacity (float): lower bound
      2nd parameter for arc capacity (float): upper bound
    2 for EXPONENTIAL
      parameter for lambda



The followings is an example:

2            This batch file generates two networks
network1     output file name of the first network
1            two-way arcs
12345678     random number seed
500          # of nodes
# this is a comment line
50           grid width
50           # of source ndoes
30           # of sink nodess
8            average degree
100000       total flow
1            distribution of arc costs, 1 for UNIFORM
0            1st parameter for arc cost (float): lower bound
2000         2nd parameter for arc cost (float): upper bound
1            distribution of arc capacities, 1 for UNIFORM
20           1st parameter for arc capacity (float): lower bound
2000         2nd parameter for arc capacity (float): upper bound


network2     output file name of the second network
0            no two-way arcs
87654321     random number seed
500          # of nodes
# this is a comment line
50           grid width
50           # of source ndoes
30           # of sink nodess
8            average degree
100000       total flow
1            distribution of arc costs, 1 for UNIFORM
0            1st parameter for arc cost (float): lower bound
2000         2nd parameter for arc cost (float): upper bound
1            distribution of arc capacities, 1 for UNIFORM
20           1st parameter for arc capacity (float): lower bound
2000         2nd parameter for arc capacity (float): upper bound

# End of batch file


**** If the parameters are read from standard input, all input items
are the same as the batch file input EXCEPT no "output file name" is
needed, and no comments are allowed.


Bugs and comments: Please direct bugs and comments to yusin@athena.mit.edu

********************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>

/*--------------------------   definitions    --------------------------*/

#define SUCCESS       1
#define FAIL          0

#define UNIFORM       1    /* uniform distribution */
#define EXPONENTIAL   2    /* exponential distribution */

/*---------------------------   structures   ---------------------------*/

struct stat_para {    /* structure for statistical distributions */
              int distribution;    /* the distribution */
              float parameter[5];  /* the parameters of the distribution */
                };

struct arcs {
              int from,         /* the FROM node of that arc */
                  to,           /* the TO node of that arc */
                  cost,         /* original cost of that arc */
                  u;            /* capacity of the arc */
	    };

struct imbalance {
              int node,		/* Node ID */
	          supply;	/* Supply of that node */
	    };

/*---------------------------   functions    ---------------------------*/

void assign_capacities(),	/* Assign a capacity to each arc */
     assign_costs(),		/* Assign a cost to each arc */
     assign_imbalance(),	/* Assign an imbalance to each node */
     err(),			/* Error handler */
     frees(),			/* Free all allocated memory */
     gen_more_arcs(),		/* generate more arcs to meet the specified
				   density */
     generate(),		/* Generate a random network */
     input(),			/* Read the control parameters */
     output(),			/* Output the network */
     select_source_sinks();	/* Randomly select the source nodes and sink
				   nodes */
char *read_line();		/* Read a line from the batch file */
int  exponential(),		/* Returns an exponentially distributed
				   integer with parameter lambda */
     initialization(),		/* Preparation work before generating the
				   network */
     int_reader(),		/* Read an integer from the input device */
     uniform();			/* Generates an integer uniformly selected
				   between specified upper and lower bound */
float float_reader(),		/* Read a float from the input device */
     randy();			/* Returns a random number between 0.0 and
				   1.0. */
struct arcs *gen_basic_grid(),	/* Generate the basic grid */
     *gen_additional_arcs();	/* generate an arc from each source to the
				   supernode and from supernode to each sink */

/*------------------------   globol variables   ------------------------*/

int seed,              /* random number seed */
    seed_original,     /* The original seed from input */
    two_way,           /* 0:generate arcs in both direction for the basic
                          grid, except for
                          the arcs to/from the super node.  1:o/w */
    n_node,            /* total number of nodes in the network, numbered
                          1 to n_node, including the super node, which is
                          the last one */
    n_arc,             /* total number of arcs in the network, counting
                          EVERY arc. */
    n_grid_arc,        /* number of arcs in the basic grid, including
                          the arcs to/from the super node */
    n_source, n_sink,  /* number of source and sink nodes */
    avg_degree,        /* average degree, arcs to and from the super node
                          are counted */
    supply,            /* total supply in the network */
    n1, n2;            /* the two edges of the network grid.  n1 >= n2 */

int parameters[]={0,2,1};  /* number of parameters of various distributions */

struct imbalance *source_list=NULL, *sink_list=NULL;
                       /* head of the array of source/sink nodes */

FILE *outfile;         /* output file */

struct stat_para
    arc_costs,         /* the distribution of arc costs */
    capacities;        /* distribution of the capacities of the arcs */
struct arcs *arc_list=NULL; /* head of the arc list array.  Arcs in this
			       array are in the order of  grid_arcs,
			       arcs to/from super node, and other arcs */

/*------------------------------ program -------------------------------*/
main(argc,argv)
  int argc;
  char **argv;
{
  int batch_loop=1;  /* number of jobs in batch */

  while (batch_loop--)
  {
    if (initialization(argc,argv,&batch_loop))
    {
      generate();		/* Generate a random network */
      output();			/* Output the network */
      frees();			/* Free all allocated memory */
    }
  }
}
/*----------------------------------------------------------------------*/
/* Assign a capacity to each arc */
void assign_capacities()
{
  struct arcs *arc_ptr = arc_list;
  int (*random)(),		/* Pointer to the appropriate random number
				   generating function */
      i;

  /* Determine the random number generator to use */
  switch (arc_costs.distribution)
  {
    case UNIFORM: random = uniform;
                  break;
    case EXPONENTIAL: random = exponential;
                  break;
  }

  /* Assign capacities to grid arcs */
  for (i=n_source+n_sink;i<n_grid_arc;i++,arc_ptr++)
  {
    arc_ptr->u = random(capacities.parameter);
  }
  i = i - n_source - n_sink;

  /* Assign capacities to arcs to/from supernode */
  for (;i<n_grid_arc;i++,arc_ptr++)
  {
    arc_ptr->u = supply;
  }

  /* Assign capacities to all other arcs */
  for (;i<n_arc;i++,arc_ptr++)
    arc_ptr->u = random(capacities.parameter);
}
/*----------------------------------------------------------------------*/
/* Assign a cost to each arc */
void assign_costs()
{
  struct arcs *arc_ptr = arc_list;
  int (*random)(),		/* Pointer to the appropriate random number
				   generating function */
      i,
      high_cost,		/* A high cost assigned to arcs to/from
				   the supernode */
      max_cost = 0;		/* The maximum cost assigned to arcs in
				   the base grid */

  /* Determine the random number generator to use */
  switch (arc_costs.distribution)
  {
    case UNIFORM: random = uniform;
                  break;
    case EXPONENTIAL: random = exponential;
                  break;
  }

  /* Assign costs to arcs in the base grid */
  for (i=n_source+n_sink;i<n_grid_arc;i++,arc_ptr++)
  {
    arc_ptr->cost = random(arc_costs.parameter);
    if (max_cost < arc_ptr->cost) max_cost = arc_ptr->cost;
  }
  i = i - n_source - n_sink;

  /* Assign costs to arcs to/from the super node */
  high_cost = max_cost * 2;
  for (;i<n_grid_arc;i++,arc_ptr++)
  {
    arc_ptr->cost = high_cost;
  }

  /* Assign costs to all other arcs */
  for (;i<n_arc;i++,arc_ptr++)
    arc_ptr->cost = random(arc_costs.parameter);
}
/*----------------------------------------------------------------------*/
/* Assign an imbalance to each node */
void assign_imbalance()
{
  int total, i;
  float avg;
  struct imbalance *ptr;

  /* assign the supply nodes */
  avg = 2.0*supply/n_source;
  do
  {
    for (i=1,total=supply,ptr=source_list+1;i<n_source;i++,ptr++)
    {
      ptr->supply = (int)(randy()*avg + 0.5);
      total -= ptr->supply;
    }
    source_list->supply = total;
  } while (total <= 0);  /* redo all if the assignment "overshooted" */

  /* assign the demand nodes */
  avg = -2.0*supply/n_sink;
  do
  {
    for (i=1,total=supply,ptr=sink_list+1;i<n_sink;i++,ptr++)
    {
      ptr->supply = (int)(randy()*avg - 0.5);
      total += ptr->supply;
    }
    sink_list->supply = -total;
  } while (total <= 0);
}
/*----------------------------------------------------------------------*/
/* Error handler */
void err(n)
  int n;
{
  switch (n)
  {
    case 1: printf("Wrong number of arguments.\n");
            exit(1);
    case 2: printf("Error in reading input file.\n");
            exit(1);
    case 3: printf("Cannot open output file, job skipped.\n");
            break;
    case 4: printf("Unable to allocate memory.\n");
            break;
    case 5: printf("Batch file error.\n");
            exit(1);
  }
}
/*----------------------------------------------------------------------*/
/* Returns an "exponentially distributed" integer with parameter lambda */
int exponential(lambda)  
  float *lambda;
{
  return ((int)(- *lambda * log((double)randy()) + 0.5));
}
/*----------------------------------------------------------------------*/
/* Read a float from the input device */
float float_reader(file)
  FILE *file;
{
  char buffer[100];
  float x;

  if (file == NULL)		/* Read from stdin */
  {
    scanf("%f",&x);
  }
  else				/* Read from batch file */
  {
    sscanf(read_line(file, buffer), "%f", &x);
  }
  return (x);
}
/*----------------------------------------------------------------------*/
/* Free all allocated memory */
void frees()
{
  if (arc_list != NULL) free(arc_list);
  if (source_list != NULL) free(source_list);
  if (sink_list != NULL) free(sink_list);

  arc_list = NULL;
  source_list = NULL;
  sink_list = NULL;
}
/*----------------------------------------------------------------------*/
/* generate an arc from each source to the supernode and from supernode to
   each sink */
struct arcs *gen_additional_arcs(arc_ptr)
  struct arcs *arc_ptr;
{
  int i;

  for (i=0;i<n_source;i++,arc_ptr++)
  {
    arc_ptr->from = source_list[i].node;
    arc_ptr->to = n_node;
  }
  for (i=0;i<n_sink;i++,arc_ptr++)
  {
    arc_ptr->to = sink_list[i].node;
    arc_ptr->from = n_node;
  }
  return (arc_ptr);
}
/*----------------------------------------------------------------------*/
/* Generate the basic grid */
struct arcs *gen_basic_grid(arc_ptr)
  struct arcs *arc_ptr;
{
  int direction = 1,
      i, j, k;

  if (two_way)			/* Generate an arc in each direction */
  {
    for (i=1;i<n_node;i += n1)
      for (j=i,k=j+n1-1;j<k;j++)
      {
        arc_ptr->from = j;
        arc_ptr->to = j + 1;
        arc_ptr++;
        arc_ptr->from = j + 1;
        arc_ptr->to = j;
        arc_ptr++;
      }
    for (i=1;i<=n1;i++)
      for (j=i+n1;j<n_node;j+=n1)
      {
        arc_ptr->from = j;
        arc_ptr->to = j - n1;
        arc_ptr++;
        arc_ptr->from = j - n1;
        arc_ptr->to = j;
        arc_ptr++;
      }
  }
  else				/* Generate one arc in each direction */
  {
    for (i=1;i<n_node;i += n1)
    {
      if (direction == 1) j = i;
      else j = i + 1;
      for (k=j+n1-1;j<k;j++)
      {
        arc_ptr->from = j;
        arc_ptr->to = j + direction;
        arc_ptr++;
      }
      direction = -direction;
    }
    for (i=1;i<=n1;i++)
    {
      j = i + n1;
      if (direction == 1)
      for (;j<n_node;j+=n1)
      {
        arc_ptr->from = j - n1;
        arc_ptr->to = j;
        arc_ptr++;
      }
      else
      for (;j<n_node;j+=n1)
      {
        arc_ptr->from = j - n1;
        arc_ptr->to = j;
        arc_ptr++;
      }
      direction = -direction;
    }
  }
  return (arc_ptr);
}
/*----------------------------------------------------------------------*/
/* generate random arcs to meet the specified density */
void gen_more_arcs(arc_ptr)
  struct arcs *arc_ptr;
{
  int i;
  float ab[2];

  *ab = 0.9;
  ab[1] = n_node - 0.99;	/* upper limit is n_node-1 because
				   the supernode cannot be selected */
  for (i=n_grid_arc;i<n_arc;i++,arc_ptr++)
  {
    arc_ptr->from = uniform(ab);
    arc_ptr->to = uniform(ab);
    if (arc_ptr->from == arc_ptr->to)
    {
      arc_ptr--;
      i--;
    }
  }
}
/*----------------------------------------------------------------------*/
/* Generate a random network */
void generate()
{
  struct arcs *arc_ptr=arc_list;

  arc_ptr = gen_basic_grid(arc_ptr);
  select_source_sinks();
  arc_ptr = gen_additional_arcs(arc_ptr);
  gen_more_arcs(arc_ptr);
  assign_costs();
  assign_capacities();
  assign_imbalance();
}
/*----------------------------------------------------------------------*/
/* this function is called at the beginning of each job, and returns
   SUCCESS or FAIL */
int initialization(argc,argv,loop)  
  int argc, *loop;
  char **argv;
{
  char output_file_name[100];
  static FILE *batchfile = NULL;

  /* if command line arguments are not correct, quit the program */
  if (argc != 2 && argc != 1) err(1);

  /* If a batch file name is in the command line, and the batch file have
     not been opened yet, open it.  Then *loop will be set to the value
     in the batch file.  Otherwise, *loop will remain to be 1 as set in
     function main() */
  if (batchfile == NULL && argc == 2)
  {
    if ((batchfile=fopen(argv[1],"r"))==NULL) err(2);
    *loop = int_reader(batchfile) - 1;
  }

  /* Read from the input device */
  input(batchfile, output_file_name);

  n_arc = n_node * avg_degree;
  n_grid_arc = (two_way + 1) * ((n1 - 1)*n2 + (n2- 1)*n1) + n_source + n_sink;
  if (n_grid_arc > n_arc) n_arc = n_grid_arc;
  if ((arc_list=(struct arcs *)malloc(n_arc * sizeof(struct arcs)))==NULL
     || (source_list=(struct imbalance *)
                    malloc(n_source * sizeof(struct imbalance)))==NULL
     || (sink_list=(struct imbalance *)
                    malloc(n_sink * sizeof(struct imbalance)))==NULL)
  {
    err(4);
    return (FAIL);
  }

  /* If input by batchfile, open the output file */
  if (argc == 2)
  {
    if ((outfile=fopen(output_file_name,"w"))==NULL)
    {
      err(3);			/* Cannot open output file */
      return (FAIL);
    }
    else return (SUCCESS);
  }
  else
  {
    outfile = NULL;
    return (SUCCESS);
  }
}
/*----------------------------------------------------------------------*/
/* Read the control parameters */
void input(file, output_file_name)
  FILE *file;			/* File pointer to the batch file */
  char *output_file_name;	/* Name of the output file */
{
  int n, i;
  char buffer[100];

  if (file != NULL)
  {
    sscanf(read_line(file, buffer), "%s", output_file_name);
  }
  two_way = int_reader(file);
  if (two_way) two_way = 1;	/* If two_way is TRUE, set it to 1 */
  seed_original = seed = int_reader(file);
  n_node = int_reader(file);
  n = int_reader(file);
  n_source = int_reader(file);
  n_sink = int_reader(file);
  avg_degree = int_reader(file);
  supply = int_reader(file);
  arc_costs.distribution = int_reader(file);
  for (i=0;i<parameters[arc_costs.distribution];i++)
    *(arc_costs.parameter+i) = float_reader(file);
  capacities.distribution = int_reader(file);
  for (i=0;i<parameters[capacities.distribution];i++)
    *(capacities.parameter+i) = float_reader(file);

  /* Calculate the edge lengths of the grid according to the input */
  if (n*n >= n_node)
  {
    n1 = n;
    n2 = (float)n_node / n + 0.5;
  }
  else
  {
    n2 = n;
    n1 = (float)n_node / n + 0.5;
  }
  n_node = n1*n2 + 1;		/* Recalculate the total number of nodes and
				   plus 1 for the super node */
}
/*----------------------------------------------------------------------*/
/* Read an integer from the input device.  If read from a file, it reads a
   whole line, take the first few bytes as input data, and ignore the rest */
int int_reader(file)
  FILE *file;
{
  char buffer[100];
  int x;

  if (file == NULL)		/* Read from stdin */
  {
    scanf("%d",&x);
  }
  else				/* Read from batch file */
  {
    sscanf(read_line(file, buffer), "%d", &x);
  }
  return (x);
}
/*----------------------------------------------------------------------*/
/* Output the network in DIMACS format.
   This function is a slight modification of a piece of code provided by
   Joseph Cheriyan.  We would like to thank him for their help. */

void output()
{
  struct arcs *arc_ptr;
  struct imbalance *imb_ptr;
  int i;

  /* If no batch file used, output to stdout */
  if (outfile == NULL) outfile = stdout;

  /* output "c", "p" records */
  fprintf(outfile,"c generated by GRIDGEN\n");
  fprintf(outfile,"c seed %d\n", seed_original);
  fprintf(outfile,"c nodes %d\n", n_node);
  fprintf(outfile,"c grid size %d X %d\n", n1, n2);
  fprintf(outfile,"c sources %d sinks %d\n", n_source, n_sink);
  fprintf(outfile,"c avg. degree %d\n", avg_degree);
  fprintf(outfile,"c supply %d\n", supply);
  switch (arc_costs.distribution)
  {
    default : break;

    case UNIFORM :
      fprintf(outfile,"c arc costs: UNIFORM distr. min %d max %d\n",
	      (int)*(arc_costs.parameter+0), (int)*(arc_costs.parameter+1));
    break;

    case EXPONENTIAL :
      fprintf(outfile,"c arc costs: EXPONENTIAL distr. lambda %d\n",
	      (int)*(arc_costs.parameter+0));
    break;
  }
  switch (capacities.distribution)
  {
    default : break;

    case UNIFORM :
      fprintf(outfile,"c arc caps :  UNIFORM distr. min %d max %d\n",
	      (int)*(capacities.parameter+0), (int)*(capacities.parameter+1) );
    break;

    case EXPONENTIAL :
      fprintf(outfile,"c arc caps :  EXPONENTIAL distr. %d lambda %d\n",
	      (int)*(capacities.parameter+0) );
    break;
  }

  fprintf(outfile,"p min %d %d\n",n_node,n_arc);

  /* Output "n node supply" */
  for (i=0,imb_ptr=source_list;i<n_source;i++,imb_ptr++)
    fprintf(outfile, "n %d %d\n", imb_ptr->node, imb_ptr->supply);

  for (i=0,imb_ptr=sink_list;i<n_sink;i++,imb_ptr++)
    fprintf(outfile, "n %d %d\n", imb_ptr->node, imb_ptr->supply);

  /* Output "a from to lowcap=0 hicap cost" */
  for (i = 0,arc_ptr = arc_list;i<n_arc;i++,arc_ptr++)
  {
    fprintf(outfile,"a %d %d 0 %d %d\n",arc_ptr->from,arc_ptr->to,
                                         arc_ptr->u, arc_ptr->cost);
  }

  if (outfile != stdout) fclose(outfile);
}
/*----------------------------------------------------------------------*/
/* Returns a random number between 0.0 and 1.0.  See Ward Cheney &
   David Kincaid, "Numerical Mathematics and Computing," 2Ed, pp. 335.  */
float randy()
{
  seed = 16807*seed % 2147483647;
  if (seed < 0) seed = -seed;
  return (seed * 4.6566128752459e-10);
}
/*----------------------------------------------------------------------*/
/* Read a line from the batch file, remove all leading blanks and tabs,
   skipping comment lines and blank lines */
char *read_line(file, buffer)
  FILE *file;
  char *buffer;
{
  char *ptr;

  do
  {
    ptr = buffer;
    if (feof(file)) err(5);	/* If reach end of the batch file,
				   output error message and exit. */
    fgets(buffer,100,file);
    while((*ptr == ' ' || *ptr == '\t') && *ptr != '\n' && *ptr != '\0') ptr++;
  } while (*ptr == '#' || *ptr == '\n' || *ptr == '\0');
  /* If the line starts with an '#', or is a blank line, skip it */
  return(ptr);
}
/*----------------------------------------------------------------------*/
/* Randomly select the source nodes and sink nodes */
void select_source_sinks()
{
  int i, *int_ptr,
      *temp_list;		/* A temporary list of nodes */
  struct imbalance *ptr;
  float ab[2];			/* Parameter for random number generator */

  *ab = 0.9;
  ab[1] = n_node - 0.99;	/* upper limit is n_node-1 because
                                        the supernode cannot be selected */
  if ((temp_list=(int *)malloc(n_node*sizeof(int)))==NULL)
  {
    err(4);
    exit(1);
  }
  for (i=0,int_ptr=temp_list;i<n_node;i++,int_ptr++) *int_ptr = 0;

  /* Select the source nodes */
  for (i=0,ptr=source_list;i<n_source;i++,ptr++)
  {
    ptr->node = uniform(ab);
    if (temp_list[ptr->node] == 1) /* check for duplicates */
    {
      ptr--;
      i--;
    }
    else temp_list[ptr->node] = 1;
  }

  /* Select the sink nodes */
  for (i=0,ptr=sink_list;i<n_sink;i++,ptr++)
  {
    ptr->node = uniform(ab);
    if (temp_list[ptr->node] == 1)
    {
      ptr--;
      i--;
    }
    else temp_list[ptr->node] = 1;
  }
  free(temp_list);
}
/*----------------------------------------------------------------------*/
/* Generates an integer uniformly selected from [a[0],a[1]] */
int uniform(a)
  float *a;			/* A 2-element array that contains the upper
				   and lower bound of the random number */
{
  return ((int)((a[1] - *a)*randy() + *a + 0.5));
}
/*----------------------------------------------------------------------*/
/*---------------------------- end of program --------------------------*/

