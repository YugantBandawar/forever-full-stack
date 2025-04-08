import orderModel from "../models/orderModel.js";
import userModel from "../models/userModel.js";
import Stripe from 'stripe'


// global variables
const currency = 'inr'
const deliveryCharge = 10

// gateway initialize
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY)



// Placing orders using COD Method
const placeOrder = async (req,res) => {
    
    try {
        
        const { userId, items, amount, address} = req.body;

        const orderData = {
            userId,
            items,
            address,
            amount,
            paymentMethod:"COD",
            payment:false,
            date: Date.now()
        }

        const newOrder = new orderModel(orderData)
        await newOrder.save()

        await userModel.findByIdAndUpdate(userId,{cartData:{}})

        res.json({success:true,message:"Order Placed"})


    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }

}

// Placing orders using Stripe Method
const placeOrderStripe = async (req,res) => {
    try {
        
        const { userId, items, amount, address} = req.body
        const { origin } = req.headers;

        const orderData = {
            userId,
            items,
            address,
            amount,
            paymentMethod:"Stripe",
            payment:false,
            date: Date.now()
        }

        const newOrder = new orderModel(orderData)
        await newOrder.save()

        const line_items = items.map((item) => ({
            price_data: {
                currency:currency,
                product_data: {
                    name:item.name
                },
                unit_amount: item.price * 100
            },
            quantity: item.quantity
        }))

        line_items.push({
            price_data: {
                currency:currency,
                product_data: {
                    name:'Delivery Charges'
                },
                unit_amount: deliveryCharge * 100
            },
            quantity: 1
        })

        const session = await stripe.checkout.sessions.create({
            success_url: `${origin}/verify?success=true&orderId=${newOrder._id}`,
            cancel_url:  `${origin}/verify?success=false&orderId=${newOrder._id}`,
            line_items,
            mode: 'payment',
        })

        res.json({success:true,session_url:session.url});

    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }
}

// Verify Stripe 
const verifyStripe = async (req,res) => {

    const { orderId, success, userId } = req.body

    try {
        if (success === "true") {
            await orderModel.findByIdAndUpdate(orderId, {payment:true});
            await userModel.findByIdAndUpdate(userId, {cartData: {}})
            res.json({success: true});
        } else {
            await orderModel.findByIdAndDelete(orderId)
            res.json({success:false})
        }
        
    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }

}




// All Orders data for Admin Panel
const allOrders = async (req,res) => {

    try {
        
        const orders = await orderModel.find({})
        res.json({success:true,orders})

    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }

}

// User Order Data For Forntend
const userOrders = async (req,res) => {
    try {
        
        const { userId } = req.body

        const orders = await orderModel.find({ userId })
        res.json({success:true,orders})

    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }
}

import nodemailer from "nodemailer";
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'yabandawar@gmail.com',
    pass: 'lxvrqbxdxgkbacoa', // NOT your regular Gmail password
  },
});

const sendUpdateEmail = async (orderId, status, userEmail) => {
    const today = new Date()
  try {
    const info = await transporter.sendMail({
      from: '"Forever" Admin Team',
      to: userEmail,
      subject: "Your Forever Order has an Update",
      text:"Forever Order Update",
      html: `
        <body style="margin: 0; padding: 0; background-color: #f4f4f7; font-family: Arial, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f4f4f7; padding: 20px;">
      <tr>
        <td align="center">
          <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
            <tr>
              <td style="background-color: #4CAF50; padding: 20px; text-align: center; color: #ffffff;">
                <h1 style="margin: 0;">Order Update</h1>
              </td>
            </tr>
            <tr>
              <td style="padding: 30px; color: #333333; font-size: 16px; line-height: 1.6;">
                <p>Your order has been <strong>${status}</strong>.</p>
                <p>Thanks for ordering with us!</p>
                <p style="color: #999;">Latest updated at: <em>${today.toLocaleDateString()}</em></p>
              </td>
            </tr>
            <tr>
              <td style="background-color: #f4f4f7; padding: 20px; text-align: center; color: #888888; font-size: 12px;">
                &copy; ${new Date().getFullYear()} Forever. All rights reserved.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
      `
    });

    console.log('Message sent: %s', info.messageId);
  } catch (error) {
    console.error('Error sending email:', error);
  }
};



// update order status from Admin Panel
const updateStatus = async (req,res) => {
    try {
        const {token} = req.headers
        const { orderId, status } = req.body
        
        const order = await orderModel.findById(orderId);
        const user = await userModel.findById(order.userId);
        const userEmail = user.email;

        await orderModel.findByIdAndUpdate(orderId, { status })
        await sendUpdateEmail(orderId, status, userEmail);
        res.json({success:true,message:'Status Updated'})

    } catch (error) {
        console.log(error)
        res.json({success:false,message:error.message})
    }
}

export { verifyStripe ,placeOrder, placeOrderStripe,  allOrders, userOrders, updateStatus}