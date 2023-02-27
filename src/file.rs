use std::io::{self, Read};


pub fn take_bytes<const LEN: usize>(v: &mut impl Read) -> io::Result<[u8; LEN]> {
    let mut read = [0u8; LEN];
    v.read_exact(&mut read)?;
    Ok(read)
}

pub trait FromBytes: Sized {
    fn from_be_bytes(v: &mut impl Read) -> io::Result<Self>;
    fn from_le_bytes(v: &mut impl Read) -> io::Result<Self>;
}

macro_rules! from_x_bytes_impl {
    ($type:ty) => {
        impl FromBytes for $type {
            fn from_be_bytes(v: &mut impl Read) -> io::Result<Self> { Ok(<$type>::from_be_bytes(take_bytes(v)?)) }
            fn from_le_bytes(v: &mut impl Read) -> io::Result<Self> { Ok(<$type>::from_le_bytes(take_bytes(v)?)) }
        }
    };
}

from_x_bytes_impl!(u8);
from_x_bytes_impl!(u16);
from_x_bytes_impl!(u32);
from_x_bytes_impl!(u64);
from_x_bytes_impl!(u128);

from_x_bytes_impl!(i8);
from_x_bytes_impl!(i16);
from_x_bytes_impl!(i32);
from_x_bytes_impl!(i64);
from_x_bytes_impl!(i128);

from_x_bytes_impl!(f32);
from_x_bytes_impl!(f64);

pub trait ReadExt {
    fn try_read_le<T: FromBytes>(&mut self) -> io::Result<T>;
    fn try_read_be<T: FromBytes>(&mut self) -> io::Result<T>;

    fn read_le<T: FromBytes>(&mut self) -> T { self.try_read_le().unwrap() }
    fn read_be<T: FromBytes>(&mut self) -> T { self.try_read_be().unwrap() }
}

impl<R: Read> ReadExt for R {
    fn try_read_le<T: FromBytes>(&mut self) -> io::Result<T> { T::from_le_bytes(self) }
    fn try_read_be<T: FromBytes>(&mut self) -> io::Result<T> { T::from_be_bytes(self) }
}

#[derive(thiserror::Error, Debug)]
pub enum IdxLoadError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid type")]
    InvalidType, // todo: expected and got
    
    #[error("invalid dimensions")]
    InvalidDimensions, // todo
    
    #[error("magic short at the beginning should be zero")]
    NonZeroMagic,
}

pub trait FromIdx: Sized {
    fn from_idx(dim: &[u32], type_code: u8, v: &mut impl Read) -> Result<Self, IdxLoadError>;
}

impl<T: FromIdx> FromIdx for Vec<T> {
    fn from_idx(dim: &[u32], type_code: u8, v: &mut impl Read) -> Result<Self, IdxLoadError> {
        if let Some((head, tail)) = dim.split_first() {
            let res: Result<Vec<_>, IdxLoadError> = (0..*head).map(|_| {
                T::from_idx(tail, type_code, v)
            }).collect();

            res
        } else { Err(IdxLoadError::InvalidDimensions) }
    }
}

macro_rules! from_idx_impl {
    ($type:ty, $code:literal) => {
        impl FromIdx for Vec<$type> {
            fn from_idx(dim: &[u32], type_code: u8, v: &mut impl Read) -> Result<Self, IdxLoadError> {
                const TYPE_CODE: u8 = $code;
                if TYPE_CODE != type_code { return Err(IdxLoadError::InvalidType) }
                if dim.len() > 0 {
                    let target = dim.iter().product();
                    (0..target).map(|_| Ok(v.try_read_be()?)).collect()
                } else { Err(IdxLoadError::InvalidDimensions) }
            }
        }
    };
}

from_idx_impl!(u8, 0x08);
from_idx_impl!(i8, 0x09);
from_idx_impl!(i16, 0x0B);
from_idx_impl!(i32, 0x0C);
from_idx_impl!(f32, 0x0D);
from_idx_impl!(f64, 0x0E);

pub fn try_read_idx_file<T: FromIdx>(reader: &mut impl std::io::Read) -> Result<T, IdxLoadError> {
    let magic_zero: u16 = reader.read_be();
    let magic_ty: u8 = reader.read_be();
    let magic_dim: u8 = reader.read_be();
    let sizes = (0..magic_dim).map(|_| Ok(reader.try_read_be()?)).collect::<Result<Vec<u32>, IdxLoadError>>()?;

    T::from_idx(&sizes[..], magic_ty, reader)
}
pub fn read_idx_file<T: FromIdx>(reader: &mut impl std::io::Read) -> T { try_read_idx_file(reader).unwrap() }

pub fn try_read_idx_path<T: FromIdx>(path: impl AsRef<std::path::Path>) -> Result<T, IdxLoadError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    try_read_idx_file(&mut reader)
}
pub fn read_idx_path<T: FromIdx>(path: impl AsRef<std::path::Path>) -> T { try_read_idx_path(path).unwrap() }

fn test(r: &mut impl Read) {
    let res: Vec<u8> = try_read_idx_file(r).unwrap();
}

