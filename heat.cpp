#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

constexpr double r = 0.12;
constexpr double pi = 3.141592653589793;

struct Grid
{
    int nx, ny, nz, first, count;
    int plane;
    vector<double> u, next;

    Grid(int x, int y, int z, int start, int planes)
        : nx(x), ny(y), nz(z), first(start), count(planes), plane(x*y),
          u(size_t(planes+2)*plane), next(u.size()) {}

    void init(bool eigen)
    {
        fill(u.begin(), u.end(), 0.0);
        fill(next.begin(), next.end(), 0.0);
        for (int k=1; k<=count; ++k)
            for (int y=1; y<ny-1; ++y)
                for (int x=1; x<nx-1; ++x)
                {
                    int z=first+k-1;
                    double a=double(x)/(nx-1), b=double(y)/(ny-1), c=double(z)/(nz-1);
                    double d=(a-.48)*(a-.48)+(b-.51)*(b-.51)+(c-.43)*(c-.43);
                    u[size_t(k)*plane+y*nx+x]=eigen
                        ? sin(pi*a)*sin(pi*b)*sin(pi*c) : 100.0*exp(-90*d);
                }
    }

    void row(int k, int y)
    {
        size_t start=size_t(k)*plane+y*nx;
        for (int x=1; x<nx-1; ++x)
        {
            size_t i=start+x;
            next[i]=(1-6*r)*u[i]+r*(u[i-1]+u[i+1]+u[i-nx]+u[i+nx]+u[i-plane]+u[i+plane]);
        }
    }

    void rows(int lo, int hi, bool parallel)
    {
        // Flatten z/y work so even thin slabs can keep several threads busy.
        if (lo>hi) return;
        int height=ny-2;
        if (parallel)
        {
            #pragma omp parallel for schedule(static)
            for (int j=0; j<(hi-lo+1)*height; ++j)
                row(lo+j/height,1+j%height);
        }
        else
        {
            for (int k=lo; k<=hi; ++k)
                for (int y=1; y<ny-1; ++y)
                    row(k,y);
        }
    }
};

void solve(Grid& g, int steps, int rank, int size, const string& mode)
{
    int left=rank ? rank-1 : MPI_PROC_NULL;
    int right=rank+1<size ? rank+1 : MPI_PROC_NULL;
    bool parallel=mode!="serial";
    for (int step=0; step<steps; ++step)
    {
        double* u=g.u.data();
        int p=g.plane, n=g.count;
        if (size>1 && mode=="overlap")
        {
            // MPI runs on the main thread. Interior updates do not touch incoming halos.
            MPI_Request req[4];
            MPI_Irecv(u,p,MPI_DOUBLE,left,11,MPI_COMM_WORLD,&req[0]);
            MPI_Irecv(u+size_t(n+1)*p,p,MPI_DOUBLE,right,10,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(u+p,p,MPI_DOUBLE,left,10,MPI_COMM_WORLD,&req[2]);
            MPI_Isend(u+size_t(n)*p,p,MPI_DOUBLE,right,11,MPI_COMM_WORLD,&req[3]);
            g.rows(2,n-1,true);
            MPI_Waitall(4,req,MPI_STATUSES_IGNORE);
            g.rows(1,1,true);
            if (n>1) g.rows(n,n,true);
        }
        else
        {
            if (size>1)
            {
                // Each rank owns its interior planes; the first and last slots are ghosts.
                MPI_Sendrecv(u+p,p,MPI_DOUBLE,left,10,u+size_t(n+1)*p,p,MPI_DOUBLE,right,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Sendrecv(u+size_t(n)*p,p,MPI_DOUBLE,right,11,u,p,MPI_DOUBLE,left,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            g.rows(1,n,parallel);
        }
        // The other buffer contains the new field; fixed boundary cells stay zero.
        g.u.swap(g.next);
    }
}

double measure(Grid& g, const string& mode, int rank, int ranks, int threads, int steps)
{
    omp_set_num_threads(threads);
    vector<double> samples;
    for (int rep=-1; rep<3; ++rep)
    {
        g.init(false);
        if (ranks>1) MPI_Barrier(MPI_COMM_WORLD);
        double start=MPI_Wtime();
        solve(g,rep<0 ? 4 : steps,rank,ranks,mode);
        double time=MPI_Wtime()-start;
        double slow=time;
        if (ranks>1) MPI_Reduce(&time,&slow,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (rep>=0) samples.push_back(slow);
    }
    sort(samples.begin(),samples.end());
    return samples[1];
}

double difference(const double* a, const double* b, size_t count)
{
    double error=0;
    for (size_t i=0;i<count;++i)
    {
        if (!isfinite(a[i]) || !isfinite(b[i])) return numeric_limits<double>::infinity();
        error=max(error,abs(a[i]-b[i]));
    }
    return error;
}

void printRow(const string& name, int ranks, int threads, double t, double serial, double error)
{
    double speed=serial/t;
    cout<<left<<setw(16)<<name<<right<<setw(7)<<ranks<<setw(9)<<threads
        <<fixed<<setprecision(4)<<setw(12)<<t<<setprecision(2)<<setw(11)<<speed
        <<setw(12)<<100*speed/(ranks*threads)<<scientific<<setw(15)<<error
        <<setw(9)<<(error<=1e-10 ? "PASS" : "FAIL")<<endl;
}

int main(int argc, char** argv)
{
    int provided, rank, ranks;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    constexpr int x=224, y=224, z=226, steps=42;
    if (provided<MPI_THREAD_FUNNELED || ranks<2 || ranks>z-2)
    {
        if (!rank) cerr<<"Launch with MPI (for example: mpiexec -n 3 heat.exe)."<<endl;
        MPI_Finalize();
        return 2;
    }
    omp_set_dynamic(0);
    double serial=0;
    int ok=1;
    vector<double> reference;

    // Serial and OpenMP use the full grid on rank 0. Other ranks wait.
    if (!rank)
    {
        cout<<"3D HEAT DIFFUSION | 224 x 224 x 226, 42 steps"<<endl;
        cout<<"Median of 3 runs; setup and correctness checks are outside the timer."<<endl<<endl;
        cout<<left<<setw(16)<<"Mode"<<right<<setw(7)<<"Ranks"<<setw(9)<<"Threads"
            <<setw(12)<<"Seconds"<<setw(11)<<"Speedup"<<setw(12)<<"Eff. %"
            <<setw(15)<<"Max error"<<setw(9)<<"Check"<<endl;
        {
            Grid g(x,y,z,1,z-2);
            serial=measure(g,"serial",0,1,1,steps);
            reference.swap(g.u);
        }
        printRow("Serial",1,1,serial,serial,0);
        Grid g(x,y,z,1,z-2);
        for (int threads : {2,4,8})
        {
            double time=measure(g,"omp",0,1,threads,steps);
            double error=difference(g.u.data(),reference.data(),reference.size());
            printRow("OpenMP",1,threads,time,serial,error);
            ok &= error<=1e-10;
        }
    }

    // Scatter only the reference used to verify each owned slab; benchmark runs exchange halos.
    int base=(z-2)/ranks, remainder=(z-2)%ranks, plane=x*y;
    vector<int> counts(ranks), offsets(ranks);
    for (int p=0;p<ranks;++p)
    {
        counts[p]=(base+(p<remainder))*plane;
        offsets[p]=(1+p*base+min(p,remainder))*plane;
    }
    Grid g(x,y,z,offsets[rank]/plane,counts[rank]/plane);
    vector<double> expected(counts[rank]);
    MPI_Scatterv(reference.data(),counts.data(),offsets.data(),MPI_DOUBLE,
                 expected.data(),counts[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);
    vector<double>().swap(reference);
    for (const string mode : {"hybrid","overlap"})
        for (int threads : {1,2})
        {
            double time=measure(g,mode,rank,ranks,threads,steps);
            double error=difference(g.u.data()+plane,expected.data(),expected.size());
            double maxError=0;
            MPI_Reduce(&error,&maxError,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            if (!rank)
            {
                printRow(mode=="hybrid" ? "MPI blocking" : "MPI overlap",ranks,threads,time,serial,maxError);
                ok &= maxError<=1e-10;
            }
        }

    // Independent analytical check: a discrete sine mode retains its shape.
    g.init(true);
    solve(g,steps,rank,ranks,"overlap");
    double lambda=1-6*r+2*r*(cos(pi/(x-1))+cos(pi/(y-1))+cos(pi/(z-1)));
    double decay=pow(lambda,steps), eigenError=0;
    for (int k=1;k<=g.count;++k)
        for (int j=0;j<y;++j)
            for (int i=0;i<x;++i)
            {
                int gz=g.first+k-1;
                double expectedValue=decay*sin(pi*i/(x-1))*sin(pi*j/(y-1))*sin(pi*gz/(z-1));
                double value=g.u[size_t(k)*plane+j*x+i];
                double error=isfinite(value) ? abs(value-expectedValue) : numeric_limits<double>::infinity();
                eigenError=max(eigenError,error);
            }
    double maxEigen=0;
    MPI_Reduce(&eigenError,&maxEigen,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (!rank)
    {
        cout<<"\nSine-mode check: max error = "<<scientific<<setprecision(3)<<maxEigen<<endl;
        ok &= maxEigen<=1e-10;
        cout<<(ok ? "PASS: all comparisons matched." : "FAIL: at least one check exceeded tolerance.")<<endl;
    }
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return ok ? 0 : 1;
}
