import React, { useContext, useEffect, useState } from 'react'
import { ShopContext } from '../context/ShopContext'
import Title from './Title';
import ProductItem from './ProductItem';
import axios from 'axios';
import { toast } from 'react-toastify';

const BestSeller = () => {
  const { backendUrl } = useContext(ShopContext);
  const [bestSeller, setBestSeller] = useState([]);

  const fetchBestSellers = async () => {
    try {
      const response = await axios.get(backendUrl + '/api/product/bestsellers');
      if (response.data.success) {
        setBestSeller(response.data.bestSellers);
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  }

  useEffect(() => {
    fetchBestSellers();
  }, []);

  return (
    <div className='my-10'>
      <div className='text-center text-3xl py-8'>
        <Title text1={'BEST'} text2={'SELLERS'} />
        <p className='w-3/4 m-auto text-xs sm:text-sm md:text-base text-gray-600'>
          These picks? Tried, tested, and totally loved. Don&apost miss our hottest bestsellers!
        </p>
      </div>

      <div className='grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 gap-y-6'>
        {bestSeller.map((item, index) => (
          item && item._id && (
            <ProductItem
              key={index}
              id={item._id}
              name={item.name}
              image={item.image}
              price={item.price}
            />
          )
        ))}
      </div>
    </div>
  )
}

export default BestSeller