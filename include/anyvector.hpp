#ifndef ANYVECTOR_HPP_
#define ANYVECTOR_HPP_

namespace llrt{

    /**
       A wrapper around a vector-like object, which can be used even
       by code that doesn't know the type of the objects in the
       vector.

       Provides only a certain subset of functions.
    */
    struct AnyVector{
        virtual void resize(size_t size)=0;
        virtual size_t size()=0;

        /**
           Move the element at fromIndex to the new index, toIndex
        */
        virtual void move(size_t fromIndex, size_t toIndex)=0;

        /**
           Replace the element at the chosen index, with a newly
           constructed element at the same index
        */
        virtual void refreshIndex(size_t index)=0;
    };

    /**
       A wrapper around an std::variant<std::vector<T1>, std::vector<T2>, ...>
       which can be used even by code that doesn't know the types of
       the vectors in the variant
     */
    struct VariantVectorWrapper{

        /**
           Call f on the AnyVector corresponding to the current vector
           held by the variant

           @param capture can be used to pass additional information
           to f, besides the AnyVector
           @param f is the function to apply
         */
        virtual void apply(void * capture, void (*f)(void *, AnyVector &))=0;
    };

    template <typename T>
    struct TVectorWrapper : public AnyVector{
        std::vector<T> &vals;
        
        TVectorWrapper(std::vector<T> &vals) : vals(vals){}
        
        virtual void resize(size_t size){vals.resize(size);}
        
        virtual size_t size(){return vals.size();}

        virtual void move(size_t fromIndex, size_t toIndex){vals.at(toIndex) = std::move(vals.at(fromIndex));}

        virtual void refreshIndex(size_t index){vals.at(index) = T();}
    };


}

#endif
