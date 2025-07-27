import React, { useContext, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { ShopContext } from '../context/ShopContext';
import { assets } from '../assets/assets';

const Navbar = () => {
  const [visible, setVisible] = useState(false);

  const { setShowSearch, getCartCount, navigate, token, setToken, setCartItems } = useContext(ShopContext);

  const logout = () => {
    navigate('/login');
    localStorage.removeItem('token');
    setToken('');
    setCartItems({});
  };

  return (
    <div className="flex items-center justify-between px-4 py-4 bg-white shadow-md sticky top-0 z-50 font-medium">
      {/* Logo (Text-based Fancy Brand Name) */}
      <Link to="/" className="text-2xl font-extrabold italic text-indigo-600 tracking-wide">
        StyleSphere
      </Link>

      {/* Desktop Nav Links */}
      <ul className="hidden sm:flex gap-8 text-sm text-gray-800 font-medium">
        {[
          { label: 'HOME', path: '/' },
          { label: 'COLLECTION', path: '/collection' },
          { label: 'ABOUT', path: '/about' },
          { label: 'CONTACT', path: '/contact' },
        ].map(({ label, path }, idx) => (
          <NavLink
            key={idx}
            to={path}
            className={({ isActive }) =>
              `relative pb-1 transition duration-300 ${
                isActive ? 'text-indigo-600 font-semibold' : 'text-gray-700 hover:text-indigo-600'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </ul>

      {/* Icons Section */}
      <div className="flex items-center gap-5 text-gray-700">
        <img
          onClick={() => {
            setShowSearch(true);
            navigate('/collection');
          }}
          src={assets.search_icon}
          className="w-5 cursor-pointer hover:scale-110 transition"
          alt="search"
        />

        <div className="relative group">
          <img
            onClick={() => (token ? null : navigate('/login'))}
            className="w-5 cursor-pointer hover:scale-110 transition"
            src={assets.profile_icon}
            alt="profile"
          />
          {token && (
            <div className="absolute right-0 mt-2 hidden group-hover:block bg-white border border-gray-200 rounded-md shadow-lg w-44 py-2 text-sm">
              <p className="px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 cursor-pointer">My Profile</p>
              <p onClick={() => navigate('/orders')} className="px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 cursor-pointer">Orders</p>
              <p onClick={logout} className="px-4 py-2 text-gray-700 hover:bg-indigo-50 hover:text-indigo-600 cursor-pointer">Logout</p>
            </div>
          )}
        </div>

        <Link to="/cart" className="relative">
          <img src={assets.cart_icon} className="w-5 hover:scale-110 transition" alt="cart" />
          <p className="absolute right-[-6px] bottom-[-6px] w-4 h-4 bg-indigo-600 text-white text-[9px] rounded-full flex items-center justify-center">
            {getCartCount()}
          </p>
        </Link>

        {/* Mobile menu icon */}
        <img
          onClick={() => setVisible(true)}
          src={assets.menu_icon}
          className="w-5 sm:hidden cursor-pointer"
          alt="menu"
        />
      </div>

      {/* Sidebar menu for small screens */}
      <div className={`fixed top-0 right-0 h-full bg-white z-50 shadow-lg transform transition-transform duration-300 ${visible ? 'w-4/5' : 'translate-x-full'} sm:hidden`}>
        <div className="flex flex-col h-full">
          <div
            onClick={() => setVisible(false)}
            className="flex items-center gap-3 p-4 border-b cursor-pointer"
          >
            <img className="h-4 rotate-180" src={assets.dropdown_icon} alt="back" />
            <p className="text-gray-700">Back</p>
          </div>
          {[
            { label: 'HOME', path: '/' },
            { label: 'COLLECTION', path: '/collection' },
            { label: 'ABOUT', path: '/about' },
            { label: 'CONTACT', path: '/contact' },
          ].map(({ label, path }, idx) => (
            <NavLink
              key={idx}
              onClick={() => setVisible(false)}
              className="px-6 py-4 border-b text-gray-700 hover:bg-indigo-50 hover:text-indigo-600"
              to={path}
            >
              {label}
            </NavLink>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
