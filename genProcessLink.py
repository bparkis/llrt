#!/usr/bin/python


# debug=True uses safe access .at() instead of [], so that if you're writing a new LinkType,
# out of bounds access will be noticed.  Slightly slower.
debug = False

import os
import sys
import argparse

def at(var):
    if debug:
        return ".at(" + var + ")"
    return "[" + var + "]"

# argument types of the kernel
def argtype(n,specifier):
    if specifier == "Ni" or specifier == "Ei" or specifier == "g" or specifier == "f" or specifier == "ni" or specifier == "ei":
        return None
    return "        using arg{0} = typename std::decay_t<typename Traits::template argument<{0}>::type>;".format(str(n))

def vec(n,specifier):
    if specifier == "Ni" or specifier == "Ei" or specifier == "f" or specifier == "ni" or specifier == "ei":
        return None
    s = "            "
    # if specifier == "e" or specifier == "n":
    #     s += "const "
    return s + "std::vector<arg{0}> & v{1};".format(str(n), specifier)

def call(specifiers):
    s = "                k("
    for n,specifier in enumerate(specifiers):
        if specifier == "Ni":
            s += "near"
        elif specifier == "Ei":
            s += "near_link"
        elif specifier == "ni":
            s += "far"
        elif specifier == "ei":
            s += "far_link"
        elif specifier == "N":
            s += "v"+specifier+at("near")
        elif specifier == "E":
            s += "v"+specifier+at("near_link")
        elif specifier == "e":
            s += "ve" + at("far_link")
        elif specifier == "n":
            s += "vn" + at("far")
        elif specifier == "f":
            s += "edgeInfo"
        elif specifier == "r":
            s += "r"
        if n < len(specifiers)-1:
            s += ", "
    s += ");"
    return s

def pkParam(n, specifier):
    if specifier == "Ni" or specifier == "Ei" or specifier == "f" or specifier == "ni" or specifier == "ei":
        return None
    if specifier == "r":
        return "link.ends[0].c.net.rng" 
    s = "link.template "
    if specifier[0] == "N" or specifier[0] == "n":
        s += "comp"
    else:
        s += "link"
    s += "Data"
    s += "<arg{0}>(".format(str(n))
    if specifier == "n" or specifier == "e":
        s += "1-whichEnd)"
    else:
        s += "whichEnd)"
    return s

	# link.template compData<arg0>(whichEnd)
	# link.template compDataNext<arg1>(whichEnd)
	# link.template linkData<arg2>(whichEnd)
	# link.template linkDataNext<arg3>(whichEnd)

def filterVariant(specifier):
    if specifier == "N":
        return "link.ends[whichEnd].c.data.values"
    if specifier == "n":
        return "link.ends[1-whichEnd].c.data.values"
    if specifier == "E":
        return "link.ends[whichEnd].data.values"
    if specifier == "e":
        return "link.ends[1-whichEnd].data.values"
    return None

def filterParam(n, specifier):
    if specifier == "N" or specifier == "n" or specifier == "E" or specifier == "e":
        return """        if (!std::holds_alternative<std::vector<arg{0}> >({1}))
            return 0;""".format(str(n), filterVariant(specifier))
    return None

def processLink(specifiers):
    argtypes = '\n'.join([argtype(n,specifiers[n]) for n in range(len(specifiers)) if argtype(n,specifiers[n])])
    filtertypes = '\n'.join([filterParam(n, specifiers[n]) for n in range(len(specifiers)) if filterParam(n, specifiers[n])])
    # ensure that r is the last PureKernel member, so it doesn't take an initialization spot
    # that should go to a vector member variable or k
    vecspec = [s for s in specifiers if s != 'r']
    vecs = '\n'.join([vec(n,vecspec[n]) for n in range(len(vecspec)) if vec(n,vecspec[n])])
    if 'r' in specifiers:
        vecs += "\n            ThreadsafeRNG r;"
    vecs += "\n            _" + "".join(specifiers) + "Kernel k;"

    pkParams = '\n'.join(["            " + pkParam(n,specifiers[n]) + "," for n in range(len(specifiers)) if pkParam(n,specifiers[n])])
    vecs_ref = vecs.replace("Kernel k", "Kernel &k")
    s="""

    template<typename TL, typename {0}Kernel, typename C=NOpT3>
    size_t ProcessLink_{0}(Link<TL> & link, int whichEnd, {0}Kernel && k, JobOptions<C> opts=NullJobOptions){{
        using Traits = function_traits<{0}Kernel>;
{1}
        using _{0}Kernel = std::remove_reference<{0}Kernel>::type;
        struct PureKernel{{
{2}
            inline void operator()(const size_t near, const size_t near_link, const size_t far, const size_t far_link, const size_t edgeInfo){{
{3}
            }}
        }};
        PureKernel pk{{
{4}
            k
        }};
        struct PureKernel_Ref{{
{5}
            inline void operator()(const size_t near, const size_t near_link, const size_t far, const size_t far_link, const size_t edgeInfo){{
{3}
            }}
        }};
        PureKernel_Ref pk_ref{{
{4}
            k
        }};
        auto li = [=,&link](PureKernel &pk, size_t start, size_t end){{
            ProcessLink(link, whichEnd, pk, start, end
#ifdef DEBUG_OP_LEVEL
                ,opts.kernelName
#endif

            );}};
        return QueueProcessLink(link, whichEnd, k, pk, pk_ref, li, opts);
    }}

""".format("".join(specifiers), argtypes + '\n' + filtertypes, vecs, call(specifiers), pkParams, vecs_ref)
    return s

def processCmp(specifiers):
    s = """
    template<typename TL, typename {0}Kernel, typename C=NOpT3> 
    size_t ProcessCmp_{0}(Component<TL> & c, {0}Kernel && k, JobOptions<C> opts= NullJobOptions){{
        return ProcessLink_{0}(*c.selfLink, 0, k, opts);
    }}
""".format("".join(specifiers))
    return s

def processLinks(specifiers):
    s = """
    template<typename TL, typename {0}Kernel, typename C=NOpT3>
    size_t ProcessLinks_{0}(std::vector<LinkEnd<TL> *> &links, {0}Kernel &&k, JobOptions<C> opts=NullJobOptions){{
        bool endOfBatch = opts.endOfBatch;
        opts.endOfBatch = false;
        bool blocking = opts.blocking;
        opts.blocking = false;
        size_t clientBatchNum=0;
        for(size_t i=0; i < links.size(); i++){{
            LinkEnd<TL> *end = links[i];
            if(i == links.size()-1){{
                opts.endOfBatch = endOfBatch;
                opts.blocking = blocking;
            }}
            size_t result = ProcessLink_{0}(end->l, end->whichEnd, k, opts);
            if (result > clientBatchNum)
                clientBatchNum = result;
        }}
        return clientBatchNum;
    }}
""".format("".join(specifiers))
    return s
    
def processNetCmps(specifiers):
    s = """
    template<typename TL, typename {0}Kernel, typename C=NOpT3> 
    size_t ProcessNetCmps_{0}(Network<TL> & net, {0}Kernel && k, JobOptions<C> opts= NullJobOptions){{
        std::vector<LinkEnd<TL> *> links;
        for(const std::unique_ptr<Component<TL>> & c : net.components){{
            if constexpr(!std::is_same_v<decltype(opts.cmpNearFilter), NullOptionType>){{
                if(!opts.cmpNearFilter(*c))
                    continue;
            }}
            links.push_back(&c->selfLink->ends[0]);
        }}
        return ProcessLinks_{0}(links, k, opts);
    }}
""".format("".join(specifiers))
    return s


def processNetLinks(specifiers):
    s = """
    template<typename TL, typename {0}Kernel, typename C=NOpT3>
    size_t ProcessNetLinks_{0}(Network<TL> & net, {0}Kernel && k,JobOptions<C> opts=NullJobOptions){{
        std::vector<LinkEnd<TL> *> links;
        for(const std::unique_ptr<Component<TL>> & c : net.components){{
            if constexpr(!std::is_same_v<decltype(opts.cmpNearFilter), NullOptionType>){{
                if(!opts.cmpNearFilter(*c))
                    continue;
            }}
            for(int i : {{0, 1}})
                for(auto &l : c->links[i])
                    if((!opts.onlyAxons || l->ends[i].isAxon())
                       && (!opts.onlyDendrites || l->ends[i].isDendrite())){{
                        if constexpr(!std::is_same_v<decltype(opts.cmpFarFilter), NullOptionType>){{
                            if(!opts.cmpFarFilter(l->ends[i].c))
                                continue;
                        }}
                        links.push_back(&l->ends[i]);
                    }}
        }}
        return ProcessLinks_{0}(links, k, opts);
    }}
""".format("".join(specifiers))
    return s

allSpecifiers = ["N","Ni","E","Ei","e","ei","n","ni","f","r"]

# supports specifiers of length up to 2
def parseSpecString(specstring):
    specifier = ""
    specifiers = []
    for i,char in enumerate(specstring):
        specifier += char
        
        if specifier in allSpecifiers:
            if i+1 < len(specstring):
                nextChar = specstring[i+1]
                if specifier + nextChar in allSpecifiers:
                    continue
            specifiers.append(specifier)
            specifier = ""
    if specifier and not specifier in allSpecifiers:
        return None # invalid specstring
    return specifiers


def genHeader(filename, generable, verbose):
    functions = ""

    with open(filename, "w") as f:
        f.write("""
#ifndef PROCESSLINK_HPP_
#define PROCESSLINK_HPP_
#include "network.hpp"

// Generated by genProcessLink.py

// Kernel parameter specifiers:
// N = near component
// Ni = index at near component
// E = near link end
// Ei = index at near link end
// n = far component
// ni = index at far component
// e = far link end
// ei = index at far link end
// f = edgeInfo
// r = ThreadsafeRNG

namespace llrt{
""")
        for (functionName, specstring) in generable:
            parsed = parseSpecString(specstring)
            if parsed:
                if verbose:
                    print("Generating " + functionName + specstring)
                if functionName == "ProcessLink_":
                    f.write(processLink(parsed))
                elif functionName == "ProcessCmpLinks_":
                    f.write(processCmpLinks(parsed))
                elif functionName == "ProcessNetLinks_":
                    f.write(processNetLinks(parsed))
                elif functionName == "ProcessCmp_":
                    f.write(processCmp(parsed))
                elif functionName == "ProcessNetCmps_":
                    f.write(processNetCmps(parsed))
                elif functionName == "ProcessLinks_":
                    f.write(processLinks(parsed))
                else:
                    if verbose:
                        print("Function " + functionName + specstring + " not yet supported")
            else:
                if verbose:
                    print("Invalid specstring for " + functionName + specstring)
        f.write("""
}
#endif
""")

import glob

specifierChars = ''.join(set("".join(allSpecifiers)))
functionNames = ["ProcessLink_", "ProcessCmpLinks_", "ProcessNetLinks_", "ProcessCmp_", "ProcessNetCmps_", "ProcessLinks_"]

def checkLine(functionName, line, verbose):
    ix=0
    results = []
    while True:
        ix = line.find(functionName, ix)
        if ix == -1:
            break
        ix2 = ix + len(functionName)
        while ix2 < len(line) and line[ix2] in specifierChars:
            ix2 += 1
        specifiers = line[ix + len(functionName):ix2]
        if verbose:
            print("  ",functionName + specifiers)
        results.append((functionName, specifiers))
        ix = ix2
    return results

def scrapeFiles(filenames, verbose):
    results = []
    for fname_glob in filenames:
        fnames = glob.glob(fname_glob)
        for fname in fnames:
            if not os.path.isfile(fname):
                continue
            with open(fname, 'r') as f:
                if verbose:
                    print ("Scraping " + fname)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    for functionName in functionNames:
                        results.extend(checkLine(functionName, line, verbose))
    extraResults = []
    for s in results:
        if s[0] in ["ProcessCmpLinks_", "ProcessNetLinks_", "ProcessCmp_", "ProcessNetCmps_"]:
            extraResults.append(("ProcessLinks_", s[1]))
            extraResults.append(("ProcessLink_", s[1]))
        if s[0] in ["ProcessLinks_"]:
            extraResults.append(("ProcessLink_", s[1]))
            
    results = list(set(results + extraResults))
    def keyPL(x):
        if x[0] == "ProcessLink_":
            return 0
        if x[0] == "ProcessLinks_":
            return 1
        return 2
    results.sort(key=keyPL) # place ProcessLink operations at the beginning so others can refer to them
    return results

def main(args):
    if os.path.isfile(args.sources_list):
        with open(args.sources_list, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                args.filesToScrape.append(line.strip())

    if len(args.filesToScrape) == 0:
        print("Nothing to scrape!")
        parser.print_help()
        exit()


    generable = scrapeFiles(args.filesToScrape, args.verbose)

    genHeader(args.targetfile, generable, args.verbose)
    

if __name__ == "__main__":
    filename = "include/process_link.hpp"
    
    parser = argparse.ArgumentParser(description="Generate a header with the ProcessLink_*, ProcessLinks_*, ProcessCmpLinks_*, ProcessNetLinks_*, ProcessCmp_*, ProcessNetCmps_* families of functions.")
    
    parser.add_argument('filesToScrape', metavar='sourcefile', type=str, nargs="*", help="a file, such as a C++ source file, to scrape for mentions of the functions to be generated  (glob syntax allowed)")
    
    parser.add_argument('-o', dest="targetfile", type=str, help="filename of the generated header (default: include/process_link.hpp)", default="include/process_link.hpp")
    
    parser.add_argument('-r', dest="sources_list", type=str, help="file containing a list of filenames of sources to scrape, one per line (glob syntax allowed) (default: .llrtfiles)", default=".llrtfiles")

    parser.add_argument('-v', dest="verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    main(args)

