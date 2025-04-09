import { v2 as cloudinary } from "cloudinary"
import productModel from "../models/productModel.js"
import orderModel from '../models/orderModel.js'

// function for add product
const addProduct = async (req, res) => {
    try {

        const { name, description, price, category, subCategory, sizes, bestseller } = req.body

        const image1 = req.files.image1 && req.files.image1[0]
        const image2 = req.files.image2 && req.files.image2[0]
        const image3 = req.files.image3 && req.files.image3[0]
        const image4 = req.files.image4 && req.files.image4[0]

        const images = [image1, image2, image3, image4].filter((item) => item !== undefined)

        let imagesUrl = await Promise.all(
            images.map(async (item) => {
                let result = await cloudinary.uploader.upload(item.path, { resource_type: 'image' });
                return result.secure_url
            })
        )

        const productData = {
            name,
            description,
            category,
            price: Number(price),
            subCategory,
            bestseller: bestseller === "true" ? true : false,
            sizes: JSON.parse(sizes),
            image: imagesUrl,
            date: Date.now()
        }

        console.log(productData);

        const product = new productModel(productData);
        await product.save()

        res.json({ success: true, message: "Product Added" })

    } catch (error) {
        console.log(error)
        res.json({ success: false, message: error.message })
    }
}

// function for list product
const listProducts = async (req, res) => {
    try {
        
        const products = await productModel.find({});
        res.json({success:true,products})

    } catch (error) {
        console.log(error)
        res.json({ success: false, message: error.message })
    }
}

// function for removing product
const removeProduct = async (req, res) => {
    try {
        
        await productModel.findByIdAndDelete(req.body.id)
        res.json({success:true,message:"Product Removed"})

    } catch (error) {
        console.log(error)
        res.json({ success: false, message: error.message })
    }
}

// function for single product info
const singleProduct = async (req, res) => {
    try {
        
        const { productId } = req.body
        const product = await productModel.findById(productId)
        res.json({success:true,product})

    } catch (error) {
        console.log(error)
        res.json({ success: false, message: error.message })
    }
}


 
 // Aggregate all the orders and idenntify the best sellers. For Best sellers only those items are considered that have been delivered and the payment is complete. That is confirmed orders. 
 
 // Iterate through each item in the items array of the order, check if the item has been delivered and the payment is made. if yes, then group same items and calculate their quantities. Finally sort them in descending order and get the top 5 items.
 const getBestSellers = async (req, res) => {
    try {
        const topSellingProducts = await orderModel.aggregate([
            { $unwind: "$items" },

            { $match: { 
                // payment: true,
                status: "Delivered"
            }},

            { $group: {
                _id: "$items._id",
                totalSold: { $sum: "$items.quantity" }
            }},

            { $sort: { totalSold: -1 }},
            { $limit: 5 }
        ]);

        const productIds = topSellingProducts.map(p => p._id);
        const bestSellers = await productModel.find({ _id: { $in: productIds }});

        const sortedBestSellers = productIds.map(id => 
            bestSellers.find(product => product._id.toString() === id)
        );

        res.json({ success: true, bestSellers: sortedBestSellers });

    } catch (error) {
        console.log(error);
        res.json({ success: false, message: error.message });
    }
}

export { listProducts, addProduct, removeProduct, singleProduct, getBestSellers }